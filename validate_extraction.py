"""Step 2c: compare LLM extraction (extract_claims.py output) against your manual
gold-standard annotation (sample_for_validation.py template, filled in by hand) and
report agreement -- this is what turns "LLM extraction" into "validated LLM
extraction" for the pitch/methods section.

Reports, per categorical field: raw agreement rate and Cohen's kappa (chance-corrected
agreement -- matters here since some categories like "central_sensitization" or
"not_specified" will dominate, and raw agreement alone overstates quality on an
imbalanced label set). For sample_size (numeric), reports exact-match rate.

Usage:
    python validate_extraction.py --corpus joint
    (expects data/validation/joint_annotation_template_completed.csv to exist --
    copy the template, fill in the empty columns by hand, save with _completed suffix)
"""

import argparse
import csv
import json

import config

CATEGORICAL_FIELDS = ["mechanism", "study_design", "comorbidity_focus", "fm_measure_type", "stance"]


def load_manual(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["pmid"]] = row
    return rows


def load_llm(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows[rec["pmid"]] = rec
    return rows


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """pairs: list of (manual_label, llm_label). Unweighted Cohen's kappa."""
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    n = len(pairs)
    if n == 0:
        return float("nan")

    idx = {label: i for i, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    for a, b in pairs:
        confusion[idx[a]][idx[b]] += 1

    po = sum(confusion[i][i] for i in range(len(labels))) / n

    row_totals = [sum(row) for row in confusion]
    col_totals = [sum(confusion[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum(row_totals[i] * col_totals[i] for i in range(len(labels))) / (n * n)

    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def interpret_kappa(k: float) -> str:
    if k != k:  # NaN
        return "undefined"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="joint")
    args = parser.parse_args()

    manual_path = config.VALIDATION_DIR / f"{args.corpus}_annotation_template_completed.csv"
    llm_path = config.EXTRACTED_DIR / f"{args.corpus}_claims.jsonl"

    if not manual_path.exists():
        raise SystemExit(
            f"Missing {manual_path}. Run sample_for_validation.py, fill in the CSV by hand, "
            f"save it with the _completed suffix."
        )
    if not llm_path.exists():
        raise SystemExit(f"Missing {llm_path}. Run extract_claims.py first.")

    manual = load_manual(manual_path)
    llm = load_llm(llm_path)

    common_pmids = sorted(set(manual) & set(llm))
    missing_llm = sorted(set(manual) - set(llm))
    if missing_llm:
        print(f"Warning: {len(missing_llm)} manually-annotated PMIDs have no LLM extraction "
              f"yet (run extract_claims.py on them): {missing_llm}")

    if not common_pmids:
        raise SystemExit("No overlapping PMIDs between manual annotation and LLM extraction.")

    print(f"Validating {len(common_pmids)} papers\n")
    print(f"{'Field':<20} {'Raw agreement':<16} {'Cohen kappa':<14} {'Interpretation'}")
    print("-" * 70)

    for field in CATEGORICAL_FIELDS:
        pairs = []
        for pmid in common_pmids:
            manual_val = manual[pmid].get(field, "").strip()
            llm_val = str(llm[pmid].get(field, "")).strip()
            if not manual_val:
                continue  # skip unannotated rows
            pairs.append((manual_val, llm_val))

        if not pairs:
            print(f"{field:<20} {'no annotations':<16}")
            continue

        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        kappa = cohens_kappa(pairs)
        print(f"{field:<20} {agree:.1%} (n={len(pairs)})   {kappa:>6.3f}       {interpret_kappa(kappa)}")

    # sample_size: numeric exact-match (treat blank manual / -1 llm as "not stated")
    size_pairs = []
    for pmid in common_pmids:
        manual_val = manual[pmid].get("sample_size", "").strip()
        if not manual_val:
            continue
        try:
            manual_n = int(manual_val)
        except ValueError:
            continue
        llm_n = llm[pmid].get("sample_size", -1)
        size_pairs.append((manual_n, llm_n))

    if size_pairs:
        exact = sum(1 for m, l in size_pairs if m == l) / len(size_pairs)
        print(f"\n{'sample_size':<20} exact match: {exact:.1%} (n={len(size_pairs)})")

    print(f"\nMismatches (for error analysis):")
    for field in CATEGORICAL_FIELDS:
        for pmid in common_pmids:
            manual_val = manual[pmid].get(field, "").strip()
            llm_val = str(llm[pmid].get(field, "")).strip()
            if manual_val and manual_val != llm_val:
                print(f"  [{field}] pmid {pmid}: manual='{manual_val}' vs llm='{llm_val}'")


if __name__ == "__main__":
    main()
