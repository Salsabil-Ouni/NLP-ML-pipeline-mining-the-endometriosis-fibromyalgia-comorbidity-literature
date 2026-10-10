"""Corpus quality audit: check that every record in a corpus actually contains its
required disease-name term(s) as literal text in its title or abstract. PubMed's
[tiab] query can silently admit records via automatic term expansion (related MeSH
concepts) that don't literally contain the searched word -- confirmed on the `joint`
corpus (6/97 flagged) and again while building rag_discovery.py (a handful of
fibro.jsonl/endo.jsonl records didn't even mention their own disease). This measures
the false-positive rate for any corpus so downstream counts can be reported honestly.

Usage:
    python audit_corpus.py --corpus joint   # requires BOTH terms present
    python audit_corpus.py --corpus endo    # requires "endometriosis" present
    python audit_corpus.py --corpus fibro   # requires "fibromyalgia" present
"""

import argparse
import json

import config
from extract_claims import load_jsonl

KEYWORD_SETS = {
    "endometriosis": ["endometriosis", "endometriotic"],
    "fibromyalgia": ["fibromyalgia", "fibromyalgic"],
}

# which terms each corpus is required to literally contain
REQUIRED_TERMS_BY_CORPUS = {
    "joint": ["endometriosis", "fibromyalgia"],  # must contain BOTH
    "endo": ["endometriosis"],
    "fibro": ["fibromyalgia"],
}


def has_any(text: str, keywords: list[str]) -> bool:
    text = text.lower()
    return any(k in text for k in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="joint")
    parser.add_argument("--show", type=int, default=15, help="Max flagged examples to print to console")
    args = parser.parse_args()

    required = REQUIRED_TERMS_BY_CORPUS.get(args.corpus)
    if required is None:
        raise SystemExit(f"No required-terms rule defined for corpus '{args.corpus}' -- "
                          f"add one to REQUIRED_TERMS_BY_CORPUS.")

    records = load_jsonl(config.RAW_DIR / f"{args.corpus}.jsonl")
    if not records:
        raise SystemExit(f"No records found for corpus '{args.corpus}'.")

    flagged = []
    for rec in records:
        text = f"{rec.get('title', '')} {rec.get('abstract', '')}"
        missing = [term for term in required if not has_any(text, KEYWORD_SETS[term])]
        if missing:
            flagged.append({"pmid": rec["pmid"], "title": rec["title"], "missing": missing})

    print(f"Corpus '{args.corpus}': {len(records)} records, {len(flagged)} flagged "
          f"({len(flagged) / len(records):.1%} false-positive-suspected rate)\n")

    for f in flagged[: args.show]:
        print(f"  pmid {f['pmid']}: missing {f['missing']}")
        print(f"    {f['title']}")
    if len(flagged) > args.show:
        print(f"  ... and {len(flagged) - args.show} more (see saved file)")

    out_path = config.RAW_DIR / f"{args.corpus}_audit_flagged.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(flagged, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved flagged list to {out_path}")
    print("These are candidates to EXCLUDE from this corpus before reporting counts "
          "-- they likely entered via PubMed's automatic term expansion rather than a "
          "literal title/abstract mention.")


if __name__ == "__main__":
    main()
