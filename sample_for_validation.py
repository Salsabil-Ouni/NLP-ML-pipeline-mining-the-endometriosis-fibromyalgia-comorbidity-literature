"""Step 2b: build a fixed random sample of the joint corpus for manual gold-standard
annotation, used to validate extract_claims.py's LLM output (precision/agreement)
instead of trusting LLM extraction unvalidated.

Writes a CSV with the abstract text and empty columns for you to fill in by hand,
using the exact same categories extract_claims.py uses (see MECHANISM_KEYS etc. in
that file) so the two are directly comparable.

Usage:
    python sample_for_validation.py --corpus joint --n 20
"""

import argparse
import csv
import json
import random

import config
from extract_claims import (
    COMORBIDITY_FOCUS_KEYS,
    FM_MEASURE_TYPE_KEYS,
    MECHANISM_KEYS,
    STANCE_KEYS,
    STUDY_DESIGN_KEYS,
    load_jsonl,
)

FIELDNAMES = [
    "pmid",
    "title",
    "abstract",
    "mechanism",
    "study_design",
    "sample_size",
    "comorbidity_focus",
    "fm_measure_type",
    "stance",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="joint")
    parser.add_argument("--n", type=int, default=20, help="Sample size for manual annotation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(config.RAW_DIR / f"{args.corpus}.jsonl")
    records = [r for r in records if r.get("abstract")]
    if len(records) < args.n:
        print(f"Warning: only {len(records)} records with abstracts available, using all of them")

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    out_path = config.VALIDATION_DIR / f"{args.corpus}_annotation_template.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in sample:
            writer.writerow({
                "pmid": rec["pmid"],
                "title": rec["title"],
                "abstract": rec["abstract"],
                "mechanism": "",
                "study_design": "",
                "sample_size": "",
                "comorbidity_focus": "",
                "fm_measure_type": "",
                "stance": "",
                "notes": "",
            })

    print(f"Wrote {len(sample)} records to {out_path}")
    print("\nFill in these columns by hand, using exactly these category values:")
    print(f"  mechanism: {', '.join(MECHANISM_KEYS)}")
    print(f"  study_design: {', '.join(STUDY_DESIGN_KEYS)}")
    print(f"  sample_size: an integer, or leave blank if not stated")
    print(f"  comorbidity_focus: {', '.join(COMORBIDITY_FOCUS_KEYS)}")
    print(f"  fm_measure_type: {', '.join(FM_MEASURE_TYPE_KEYS)}")
    print(f"  stance: {', '.join(STANCE_KEYS)}")
    print(f"\nSave as {out_path.stem}_completed.csv when done, then run validate_extraction.py")


if __name__ == "__main__":
    main()
