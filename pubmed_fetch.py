"""
Fetch PubMed abstracts for endometriosis-fibromyalgia literature mining.

Uses NCBI E-utilities (esearch + efetch) directly -- free, no API key required,
though supplying one raises the rate limit from 3 req/s to 10 req/s.
NCBI requires an identifying email/tool name on every request (usage policy),
set via --email or the ENTREZ_EMAIL env var.

Corpora fetched (see corpora dict below):
  - joint:  endometriosis AND fibromyalgia co-mentions (the primary corpus --
            this is the "unfilled niche" corpus: papers that discuss both
            conditions together)
  - endo:   endometriosis literature generally (for comparison/context)
  - fibro:  fibromyalgia literature generally (for comparison/context,
            mirrors Folajimi 2025's corpus)

Output: one JSONL file per corpus under data/raw/, one record per article,
resumable via a saved PMID cursor so an interrupted run doesn't restart
from scratch or re-fetch records already saved.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

EFETCH_BATCH_SIZE = 200

CORPORA = {
    "joint": '("endometriosis"[tiab]) AND ("fibromyalgia"[tiab])',
    "endo": '"endometriosis"[tiab]',
    "fibro": '"fibromyalgia"[tiab]',
}

DATA_DIR = Path(__file__).parent / "data" / "raw"


def http_get(url, params, max_retries=5):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(full_url, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"  request failed ({e}), retrying in {wait}s "
                  f"(attempt {attempt}/{max_retries})", file=sys.stderr)
            time.sleep(wait)


ESEARCH_HARD_CAP = 9999  # NCBI esearch refuses retstart > 9998 regardless of query size


def _esearch_page(query, email, api_key, retstart, retmax, mindate=None, maxdate=None):
    """One esearch call. Returns (idlist, total_count). Retries on transient
    malformed JSON (NCBI occasionally sends a response with a stray control
    character) and raises RuntimeError on a genuine API error payload."""
    params = {
        "db": "pubmed",
        "term": query,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
        "email": email,
        "tool": "endo-fibro-lit-mining",
    }
    if mindate and maxdate:
        params["datetype"] = "pdat"
        params["mindate"] = mindate
        params["maxdate"] = maxdate
    if api_key:
        params["api_key"] = api_key

    for _ in range(3):
        raw = http_get(ESEARCH_URL, params)
        try:
            payload = json.loads(raw, strict=False)
            break
        except json.JSONDecodeError as e:
            print(f"  esearch returned malformed JSON ({e}), retrying in 5s", file=sys.stderr)
            time.sleep(5)
    else:
        raise RuntimeError("esearch: repeated malformed JSON, giving up")

    result = payload.get("esearchresult", {})
    if "ERROR" in result or "ERROR" in payload:
        raise RuntimeError(f"esearch API error: {result.get('ERROR') or payload.get('ERROR')}")
    return result.get("idlist", []), int(result.get("count", 0))


def _split_date_range(start_year: int, end_year: int):
    """Split a year range into two halves for recursive divide-and-conquer
    when a range's result count exceeds NCBI's 9999 hard cap."""
    if start_year == end_year:
        # Can't split further by year; caller must accept partial results (rare:
        # would need month-level splitting, but no single year of fibromyalgia/
        # endometriosis literature is anywhere near 9999 papers).
        return None
    mid = (start_year + end_year) // 2
    return (start_year, mid), (mid + 1, end_year)


def esearch_all_pmids(query, email, api_key, rate_delay, start_year=1945, end_year=2026):
    """Return the full list of PMIDs matching query, using date-range
    divide-and-conquer to work around NCBI's hard 9999-result cap per query --
    a single query.term search cannot page past 9999 no matter what retstart is
    used (confirmed via NCBI's own error message), so a query with more matches
    than that must be split into date ranges each individually under the cap."""
    all_pmids: set = set()

    def fetch_range(start_year, end_year):
        mindate, maxdate = f"{start_year}/01/01", f"{end_year}/12/31"
        ids, total = _esearch_page(query, email, api_key, 0, 0, mindate, maxdate)
        time.sleep(rate_delay)

        if total > ESEARCH_HARD_CAP:
            split = _split_date_range(start_year, end_year)
            if split is None:
                print(f"  [warn] {start_year}: {total} results in a single year, "
                      f"exceeds cap and can't split further -- truncating to first {ESEARCH_HARD_CAP}")
            else:
                (a1, a2), (b1, b2) = split
                fetch_range(a1, a2)
                fetch_range(b1, b2)
                return

        retstart = 0
        retmax = 9999
        while True:
            ids, total = _esearch_page(query, email, api_key, retstart, retmax, mindate, maxdate)
            all_pmids.update(ids)
            time.sleep(rate_delay)
            print(f"  esearch [{start_year}-{end_year}]: {len(all_pmids)} total PMIDs so far "
                  f"(this range: {total})")
            retstart += retmax
            if retstart >= total or not ids:
                break

    fetch_range(start_year, end_year)
    return list(all_pmids)


def parse_article(article_elem):
    """Extract fields from one PubmedArticle XML element."""
    pmid_elem = article_elem.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None else None

    title_elem = article_elem.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    abstract_parts = []
    for ab_text in article_elem.findall(".//Abstract/AbstractText"):
        label = ab_text.get("Label")
        text = "".join(ab_text.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    journal_elem = article_elem.find(".//Journal/Title")
    journal = journal_elem.text if journal_elem is not None else ""

    year = None
    for path in (".//PubDate/Year", ".//PubDate/MedlineDate"):
        y = article_elem.find(path)
        if y is not None and y.text:
            year = y.text[:4]
            break

    authors = []
    for author in article_elem.findall(".//AuthorList/Author"):
        last = author.find("LastName")
        fore = author.find("ForeName")
        if last is not None:
            name = last.text
            if fore is not None:
                name = f"{fore.text} {name}"
            authors.append(name)

    mesh_terms = []
    for mh in article_elem.findall(".//MeshHeading/DescriptorName"):
        if mh.text:
            mesh_terms.append(mh.text)

    doi = None
    for eid in article_elem.findall(".//ArticleId"):
        if eid.get("IdType") == "doi":
            doi = eid.text

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": authors,
        "mesh_terms": mesh_terms,
        "doi": doi,
    }


def efetch_batch(pmids, email, api_key, rate_delay):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": email,
        "tool": "endo-fibro-lit-mining",
    }
    if api_key:
        params["api_key"] = api_key
    raw = http_get(EFETCH_URL, params)
    time.sleep(rate_delay)
    root = ET.fromstring(raw)
    return [parse_article(a) for a in root.findall(".//PubmedArticle")]


def load_existing_pmids(out_path):
    if not out_path.exists():
        return set()
    seen = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add(rec["pmid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def fetch_corpus(name, query, email, api_key, rate_delay, limit=None):
    print(f"\n=== Corpus: {name} ===")
    print(f"Query: {query}")

    out_path = DATA_DIR / f"{name}.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    already_have = load_existing_pmids(out_path)
    if already_have:
        print(f"  resuming: {len(already_have)} records already saved")

    all_pmids = esearch_all_pmids(query, email, api_key, rate_delay)
    if limit:
        all_pmids = all_pmids[:limit]
    todo = [p for p in all_pmids if p not in already_have]
    print(f"  {len(all_pmids)} total PMIDs, {len(todo)} remaining to fetch")

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), EFETCH_BATCH_SIZE):
            batch = todo[i:i + EFETCH_BATCH_SIZE]
            print(f"  efetch batch {i // EFETCH_BATCH_SIZE + 1} "
                  f"({len(batch)} records, {i + len(batch)}/{len(todo)})")
            try:
                records = efetch_batch(batch, email, api_key, rate_delay)
            except Exception as e:
                print(f"  batch failed after retries, skipping: {e}", file=sys.stderr)
                continue
            for rec in records:
                if rec["pmid"] is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"  done. Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=os.environ.get("ENTREZ_EMAIL"),
                         help="Required by NCBI usage policy. Or set ENTREZ_EMAIL.")
    parser.add_argument("--api-key", default=os.environ.get("ENTREZ_API_KEY"),
                         help="Optional NCBI API key (raises rate limit to 10 req/s).")
    parser.add_argument("--corpus", choices=list(CORPORA.keys()) + ["all"],
                         default="joint", help="Which corpus to fetch.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap number of PMIDs fetched per corpus (for testing).")
    args = parser.parse_args()

    if not args.email:
        parser.error(
            "NCBI requires an identifying email on every request. "
            "Pass --email you@example.com or set ENTREZ_EMAIL."
        )

    rate_delay = 0.11 if args.api_key else 0.34  # stay under 10/s or 3/s

    targets = list(CORPORA.items()) if args.corpus == "all" else [(args.corpus, CORPORA[args.corpus])]
    for name, query in targets:
        fetch_corpus(name, query, args.email, args.api_key, rate_delay, args.limit)


if __name__ == "__main__":
    main()
