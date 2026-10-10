"""Step 4 (the flagship, novel piece): retrieval-augmented discovery of latent
connections between the SEPARATE endometriosis-only and fibromyalgia-only literatures
-- papers that discuss the same underlying mechanism (central sensitization,
nociplastic pain) without ever citing each other or mentioning both conditions.

This is not summarizing what's already written; it's finding connections neither
paper's authors made explicit. Pipeline:

  1. Keyword-filter endo-only and fibro-only corpora down to abstracts that discuss
     sensitization/nociplastic-pain-relevant mechanisms (keeps the search tractable --
     no need to embed the entire literature, just the mechanism-relevant slice).
  2. Exclude anything already in the `joint` corpus (those papers already made the
     connection explicit; not what we're looking for).
  3. Embed the filtered abstracts locally (sentence-transformers, free, no API) and
     find high-similarity endo/fibro pairs via cosine similarity.
  4. Drop pairs that already cite each other (checked via the free Semantic Scholar
     API) -- a cited connection is a known one, not a latent one.
  5. Send the surviving candidate pairs to an LLM (Gemini free tier) to verify: do
     these two abstracts actually describe the same mechanism, and if so, explain how.

Usage:
    python rag_discovery.py                          # full pipeline
    python rag_discovery.py --skip-citation-check     # skip step 4 (citation API can be slow/flaky)
    python rag_discovery.py --top-k 30 --threshold 0.6
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from tqdm import tqdm

import config
from extract_claims import _extract_retry_after, _extract_status_code, load_jsonl

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# characters abstracts often contain (Greek letters like beta, en-dashes, etc).
# Reconfigure stdout to UTF-8 with replacement instead of crashing mid-run --
# results are already saved to disk by the time any printing happens.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MECHANISM_KEYWORDS = [
    "central sensitization", "centrally sensitized", "nociplastic",
    "widespread pain", "amplified pain", "pain amplification",
    "neuroinflammation", "neuroplastic", "central nervous system sensitization",
    "hyperalgesia", "allodynia", "chronic overlapping pain",
]

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "same_mechanism": {"type": "boolean"},
        "shared_mechanism": {"type": "string"},
        "explanation": {
            "type": "string",
            "description": "One to two sentences explaining the connection, or why there isn't one.",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["same_mechanism", "shared_mechanism", "explanation", "confidence"],
}

VERIFY_SYSTEM_PROMPT = """You are checking whether two biomedical abstracts, one about endometriosis and one about fibromyalgia, from SEPARATE, non-citing papers, actually describe the same underlying pain/disease mechanism (e.g. central sensitization, nociplastic pain, neuroinflammation).

Be conservative: only say same_mechanism=true if both abstracts make a specific, comparable mechanistic claim, not just because both mention "pain" or "chronic illness" generally. Respond with a single JSON object matching the schema, no text outside it."""


def keyword_filter(records: list[dict]) -> list[dict]:
    pattern = re.compile("|".join(re.escape(k) for k in MECHANISM_KEYWORDS), re.IGNORECASE)
    return [r for r in records if r.get("abstract") and pattern.search(r["abstract"])]


def disease_name_filter(records: list[dict], required_terms: list[str]) -> list[dict]:
    """Keep only records whose title/abstract literally contains at least one of
    required_terms. PubMed's [tiab] query can silently admit papers via automatic
    term expansion that don't actually contain the literal disease name in the
    fetched text (confirmed via audit_corpus.py on the joint corpus, and again here
    -- some fibro.jsonl/endo.jsonl records don't mention their own disease name).
    Without this filter, such a record can appear in BOTH the fibro-only and
    endo-only corpora, causing rag_discovery to "discover" a paper connected to
    itself."""
    pattern = re.compile("|".join(re.escape(t) for t in required_terms), re.IGNORECASE)
    return [r for r in records if pattern.search(f"{r.get('title', '')} {r.get('abstract', '')}")]


def embed_shared(fibro_texts: list[str], endo_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Embed both literatures into ONE shared semantic space, so cosine similarity
    between a fibromyalgia vector and an endometriosis vector is meaningful.

    Uses TF-IDF + Truncated SVD (LSA), fit jointly on both texts combined, rather
    than sentence-transformers -- this environment's torch install fails to load
    (a Windows DLL init error, `c10.dll`, unrelated to this project's code) even
    after a clean reinstall, so a transformer-embedding backend isn't reliably
    available here. TF-IDF+SVD needs no torch, is free, and is a well-established
    semantic similarity method; the key correctness requirement is fitting one
    vectorizer/SVD on the union of both corpora, never two separate ones, since
    two independently-fit embedding spaces are not comparable by cosine similarity
    even if their dimensions happen to match.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    combined = fibro_texts + endo_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1, 2))
    X = vectorizer.fit_transform(combined)

    n_components = min(300, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(X)
    emb = normalize(emb)  # so dot product == cosine similarity, matching top_candidate_pairs' assumption

    n_fibro = len(fibro_texts)
    return emb[:n_fibro], emb[n_fibro:]


def top_candidate_pairs(fibro_recs, fibro_emb, endo_recs, endo_emb, top_k, threshold):
    sims = fibro_emb @ endo_emb.T  # cosine similarity, since embeddings are normalized
    pairs = []
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            if sims[i, j] >= threshold:
                pairs.append((sims[i, j], i, j))
    pairs.sort(key=lambda x: -x[0])
    pairs = pairs[:top_k]
    return [
        {
            "similarity": float(score),
            "fibro": fibro_recs[i],
            "endo": endo_recs[j],
        }
        for score, i, j in pairs
    ]


def semantic_scholar_related(pmid: str) -> set:
    """Return the set of PMIDs this paper cites or is cited by, via the free
    Semantic Scholar API (no key required at this volume). Returns empty set on
    any failure -- this is a best-effort filter, not a hard dependency."""
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}"
        f"?fields=citations.externalIds,references.externalIds"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "endo-fibro-lit-mining/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return set()

    related = set()
    for group in ("citations", "references"):
        for entry in data.get(group, []) or []:
            ext_ids = entry.get("externalIds") or {}
            pm = ext_ids.get("PubMed")
            if pm:
                related.add(str(pm))
    return related


def filter_already_connected(candidates: list[dict], sleep: float) -> list[dict]:
    survivors = []
    cache = {}
    for cand in tqdm(candidates, desc="checking citation overlap"):
        fibro_pmid = cand["fibro"]["pmid"]
        endo_pmid = cand["endo"]["pmid"]
        if fibro_pmid not in cache:
            cache[fibro_pmid] = semantic_scholar_related(fibro_pmid)
            time.sleep(sleep)
        if endo_pmid in cache[fibro_pmid]:
            continue  # already cites/cited-by -- known connection, skip
        survivors.append(cand)
    return survivors


def verify_with_llm(client, model: str, fibro_rec: dict, endo_rec: dict, sleep: float) -> dict:
    text = (
        f"Abstract A (fibromyalgia paper):\n{fibro_rec['title']}\n{fibro_rec['abstract'][:2000]}\n\n"
        f"Abstract B (endometriosis paper):\n{endo_rec['title']}\n{endo_rec['abstract'][:2000]}"
    )
    while True:
        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=VERIFY_SYSTEM_PROMPT,
                input=text,
                response_format={"type": "text", "mime_type": "application/json", "schema": VERIFY_SCHEMA},
            )
        except Exception as exc:
            code = _extract_status_code(exc)
            if code == 429:
                wait = _extract_retry_after(exc, default=30.0)
                print(f"  [rate limit] waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
        return json.loads(interaction.output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=30, help="Max candidate pairs to verify with the LLM")
    parser.add_argument("--threshold", type=float, default=0.55, help="Min cosine similarity to consider a candidate")
    parser.add_argument("--skip-citation-check", action="store_true", help="Skip the Semantic Scholar citation filter")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--model", default=config.GEMINI_MODEL)
    args = parser.parse_args()

    joint_pmids = {r["pmid"] for r in load_jsonl(config.RAW_DIR / "joint.jsonl")}

    fibro_all = load_jsonl(config.RAW_DIR / "fibro.jsonl")
    endo_all = load_jsonl(config.RAW_DIR / "endo.jsonl")
    if not fibro_all or not endo_all:
        raise SystemExit("Missing data/raw/fibro.jsonl or endo.jsonl -- run "
                          "pubmed_fetch.py --corpus fibro and --corpus endo first.")

    fibro_all = disease_name_filter(fibro_all, ["fibromyalgia", "fibromyalgic"])
    endo_all = disease_name_filter(endo_all, ["endometriosis", "endometriotic"])

    fibro_recs = keyword_filter([r for r in fibro_all if r["pmid"] not in joint_pmids])
    endo_recs = keyword_filter([r for r in endo_all if r["pmid"] not in joint_pmids])

    # Defensive: even after the above, exclude any pmid appearing in both lists
    # (e.g. a genuine dual-topic paper that the joint query's own [tiab] filter
    # happened to miss) -- a paper can't be evidence of a connection to itself.
    fibro_pmids_set = {r["pmid"] for r in fibro_recs}
    endo_pmids_set = {r["pmid"] for r in endo_recs}
    overlap = fibro_pmids_set & endo_pmids_set
    if overlap:
        print(f"Excluding {len(overlap)} pmids present in both single-condition corpora "
              f"(same paper, not a cross-literature connection): {sorted(overlap)}")
        fibro_recs = [r for r in fibro_recs if r["pmid"] not in overlap]
        endo_recs = [r for r in endo_recs if r["pmid"] not in overlap]

    print(f"Mechanism-relevant, non-joint, disease-name-verified: {len(fibro_recs)} fibromyalgia papers, "
          f"{len(endo_recs)} endometriosis papers")

    if not fibro_recs or not endo_recs:
        raise SystemExit("No mechanism-relevant candidates found -- fetch larger corpora first.")

    print("Embedding both literatures into a shared semantic space (TF-IDF + SVD)...")
    fibro_texts = [f"{r['title']}. {r['abstract']}" for r in fibro_recs]
    endo_texts = [f"{r['title']}. {r['abstract']}" for r in endo_recs]
    fibro_emb, endo_emb = embed_shared(fibro_texts, endo_texts)

    candidates = top_candidate_pairs(fibro_recs, fibro_emb, endo_recs, endo_emb, args.top_k, args.threshold)
    print(f"\n{len(candidates)} candidate pairs above similarity threshold {args.threshold}")

    if not args.skip_citation_check:
        candidates = filter_already_connected(candidates, args.sleep)
        print(f"{len(candidates)} pairs survive after removing already-cited connections")

    config.require_gemini_credentials()
    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    results = []
    for cand in tqdm(candidates, desc="LLM verification"):
        try:
            verdict = verify_with_llm(client, args.model, cand["fibro"], cand["endo"], args.sleep)
        except Exception as exc:
            print(f"  [error] verification failed: {exc}")
            continue
        results.append({
            "similarity": cand["similarity"],
            "fibro_pmid": cand["fibro"]["pmid"],
            "fibro_title": cand["fibro"]["title"],
            "endo_pmid": cand["endo"]["pmid"],
            "endo_title": cand["endo"]["title"],
            **verdict,
        })
        time.sleep(args.sleep)

    confirmed = [r for r in results if r["same_mechanism"]]
    out_path = config.DISCOVERY_DIR / "connections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"all_candidates": results, "confirmed_connections": confirmed}, f, ensure_ascii=False, indent=2)

    print(f"\n{len(confirmed)} / {len(results)} candidate pairs confirmed by LLM as same-mechanism connections")
    print(f"Saved to {out_path}")
    for r in sorted(confirmed, key=lambda x: -x["confidence"])[:10]:
        print(f"\n  [{r['confidence']:.2f}] {r['shared_mechanism']}")
        print(f"    FM:   {r['fibro_title']}")
        print(f"    Endo: {r['endo_title']}")
        print(f"    {r['explanation']}")


if __name__ == "__main__":
    main()
