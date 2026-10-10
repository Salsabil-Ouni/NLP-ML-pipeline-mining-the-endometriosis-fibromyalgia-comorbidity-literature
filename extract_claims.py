"""Step 2: LLM-based structured claim extraction from the joint (endometriosis +
fibromyalgia) corpus.

Turns each abstract into a structured evidence-table row: what mechanism it claims
links the two conditions (if any), study design, sample size, whether the comorbidity
is the paper's primary focus or an incidental mention, and its stance toward the
shared-mechanism hypothesis. This becomes the results table for the "joint NLP
extraction across both conditions" gap the minireview identifies -- previously nobody
extracted this structure computationally.

Uses Gemini's free tier (same pattern as fibro-nlp-pipeline/tag_posts.py) with
structured JSON output, so responses are schema-validated rather than freeform text.

Usage:
    python extract_claims.py --corpus joint
    python extract_claims.py --corpus joint --limit 10   # smoke test
    python extract_claims.py --corpus joint --retag      # ignore cache, re-extract all
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from tqdm import tqdm

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MECHANISM_KEYS = [
    "central_sensitization",
    "nociplastic_pain",
    "autoimmune",
    "psychiatric_comorbidity",
    "surgical_trigger",
    "gut_microbiome",
    "genetic_predisposition",
    "hormonal",
    "neuropathic_small_fiber",
    "other",
    "not_specified",
]

FM_MEASURE_TYPE_KEYS = [
    "diagnosed_comorbidity",   # fibromyalgia is named as a real, diagnosed condition present in (some of) the
                                # study's OWN population -- even a brief list-mention counts (e.g. "comorbidities
                                # include ... fibromyalgia ...") as long as it refers to THIS study's patients,
                                # not just background/theoretical discussion
    "survey_instrument_only",  # the Fibromyalgia Survey Score / similar is used as a continuous nociplastic-pain
                                # measure, not to diagnose or study fibromyalgia as a comorbidity
    "not_applicable",          # fibromyalgia appears ONLY in citations of other studies' findings, or in purely
                                # theoretical/background discussion -- never tied to this study's own population
]

STUDY_DESIGN_KEYS = [
    "cohort",
    "case_control",
    "cross_sectional",
    "case_report",
    "systematic_review",
    "narrative_review",
    "rct",
    "qualitative",
    "protocol",
    "other",
]

COMORBIDITY_FOCUS_KEYS = [
    "primary_focus",       # the paper's main subject IS the endo-fibro comorbidity
    "secondary_mention",   # comorbidity is discussed but not the main subject
    "incidental_comention", # both conditions just co-occur in a list/example
]

STANCE_KEYS = [
    "supports_shared_mechanism",
    "mixed_uncertain",
    "does_not_address",
    "contradicts_shared_mechanism",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanism": {"type": "string", "enum": MECHANISM_KEYS},
        "study_design": {"type": "string", "enum": STUDY_DESIGN_KEYS},
        "sample_size": {
            "type": "integer",
            "description": "Total participant/sample count if stated in the abstract, else -1.",
        },
        "comorbidity_focus": {"type": "string", "enum": COMORBIDITY_FOCUS_KEYS},
        "fm_measure_type": {"type": "string", "enum": FM_MEASURE_TYPE_KEYS},
        "stance": {"type": "string", "enum": STANCE_KEYS},
        "evidence": {
            "type": "string",
            "description": "A short span (a few words, not a full sentence) quoted verbatim from the abstract that justifies the mechanism/stance classification.",
        },
    },
    "required": ["mechanism", "study_design", "sample_size", "comorbidity_focus", "fm_measure_type", "stance", "evidence"],
}

SYSTEM_PROMPT = f"""You are extracting structured data from biomedical abstracts for a systematic literature review on the endometriosis-fibromyalgia comorbidity. For each abstract, classify:

- mechanism: the primary mechanism the paper proposes or investigates as linking endometriosis and fibromyalgia. One of: {", ".join(MECHANISM_KEYS)}. Use "not_specified" if no mechanism is discussed. Note "hormonal" (e.g. ovarian hormone influence) and "neuropathic_small_fiber" (e.g. small fibre neuropathy, nerve fibre pathology) are valid, distinct mechanisms -- do not default these to "other".
- study_design: one of: {", ".join(STUDY_DESIGN_KEYS)}.
- sample_size: total participant count as an integer if explicitly stated, else -1.
- comorbidity_focus: one of: {", ".join(COMORBIDITY_FOCUS_KEYS)}. Judge this by what the study actually MEASURES OR COMPARES, not by which words it uses. If the study's primary outcome directly compares or quantifies endometriosis-related and fibromyalgia-related pain in the same analysis (even if it uses an instrument like the Fibromyalgia Survey Score rather than a formal fibromyalgia diagnosis), that is "primary_focus". Reserve "incidental_comention" for cases where both conditions are just listed among several unrelated conditions with no direct comparison.
- fm_measure_type: one of: {", ".join(FM_MEASURE_TYPE_KEYS)}. Default to "diagnosed_comorbidity" whenever fibromyalgia is named as a condition present in the study's OWN population, EVEN IF it is only a brief mention in a list of comorbidities/adjustment variables/covariates. Use "survey_instrument_only" specifically when a fibromyalgia-derived questionnaire (e.g. Fibromyalgia Survey Score / FM Survey Score / ACR fibromyalgia criteria) is used purely as a continuous measure of nociplastic pain in patients who may not have fibromyalgia. Reserve "not_applicable" ONLY for cases where fibromyalgia appears exclusively in citations of other studies' findings or in purely theoretical/background discussion, never tied to this study's own population at all.

  Worked examples for fm_measure_type (this field is the easiest to get wrong -- a brief mention is still "diagnosed_comorbidity", not "not_applicable"):
  * "PBS/IC is often comorbid with allergies, endometriosis, fibromyalgia, irritable bowel syndrome..." -> diagnosed_comorbidity (fibromyalgia is named as a real comorbidity of the patients being discussed, even though it's just one item in a list)
  * "...adjusting for diabetes, hypertension, ... fibromyalgia, ... endometriosis, asthma..." -> diagnosed_comorbidity (fibromyalgia is a real adjustment covariate measured in this study's population, not a citation)
  * "Prior work has linked fibromyalgia to central sensitization (Smith et al., 2020)." -> not_applicable (this ONLY cites another study's finding about fibromyalgia in general; it says nothing about fibromyalgia being present in THIS study's own population)
- stance: one of: {", ".join(STANCE_KEYS)}. Base this ONLY on what the paper's results/discussion/conclusion actually claims about a shared mechanism between the two conditions specifically -- not on whether the background/introduction merely gestures at a general hypothesis, and not on co-occurrence/prevalence statistics alone unless the paper explicitly interprets them as evidence for or against a shared mechanism. Pure epidemiological association with no mechanistic interpretation should be "does_not_address", even if the introduction mentions a hypothesis.
- evidence: a short verbatim quote (a few words) from the abstract supporting your mechanism/stance classification.

Respond with a single JSON object matching the given schema. Do not include text outside the JSON object."""

_RETRY_AFTER_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _extract_status_code(exc: Exception):
    code = getattr(exc, "status_code", None)
    if code is not None:
        return code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def _extract_retry_after(exc: Exception, default: float) -> float:
    match = _RETRY_AFTER_RE.search(str(exc))
    return float(match.group(1)) + 1.0 if match else default


def call_gemini_with_retry(client, model: str, text: str, sleep: float) -> dict:
    malformed_attempts = 0
    max_malformed_attempts = 4

    while True:
        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=SYSTEM_PROMPT,
                input=f"Abstract to classify:\n\n{text}",
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": RESPONSE_SCHEMA,
                },
            )
        except Exception as exc:
            code = _extract_status_code(exc)
            if code == 429:
                wait = _extract_retry_after(exc, default=30.0)
                print(f"  [rate limit] waiting {wait:.0f}s (server-specified)")
                time.sleep(wait)
                continue
            if code is not None and code >= 500:
                print(f"  [warn] server error {code}, retrying in 10s")
                time.sleep(10.0)
                continue
            raise

        try:
            result = json.loads(interaction.output_text)
        except (json.JSONDecodeError, TypeError) as exc:
            malformed_attempts += 1
            if malformed_attempts >= max_malformed_attempts:
                raise ValueError(f"failed after {max_malformed_attempts} malformed-response retries: {exc}")
            time.sleep(sleep)
            continue

        missing = [k for k in RESPONSE_SCHEMA["required"] if k not in result]
        if missing:
            malformed_attempts += 1
            if malformed_attempts >= max_malformed_attempts:
                raise ValueError(f"failed after {max_malformed_attempts} validation retries: missing {missing}")
            time.sleep(sleep)
            continue

        return result


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="joint", help="Corpus name (matches data/raw/<corpus>.jsonl)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between successful API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending records")
    parser.add_argument("--retag", action="store_true", help="Ignore cache, re-extract everything")
    parser.add_argument("--model", default=config.GEMINI_MODEL)
    args = parser.parse_args()

    config.require_gemini_credentials()

    in_path = config.RAW_DIR / f"{args.corpus}.jsonl"
    out_path = config.EXTRACTED_DIR / f"{args.corpus}_claims.jsonl"

    records = load_jsonl(in_path)
    if not records:
        raise SystemExit(f"No records found at {in_path}. Run pubmed_fetch.py first.")

    already_done = set()
    if not args.retag:
        for rec in load_jsonl(out_path):
            already_done.add(rec["pmid"])

    pending = [r for r in records if r["pmid"] not in already_done and r.get("abstract")]
    skipped_no_abstract = sum(1 for r in records if not r.get("abstract"))
    if args.limit:
        pending = pending[: args.limit]

    print(f"{len(pending)} pending extraction ({len(already_done)} already done, "
          f"{skipped_no_abstract} skipped for empty abstract)")

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    mode = "a" if not args.retag else "w"
    if args.retag:
        # rewrite: keep records not in this run's target set untouched isn't needed here
        # since --retag re-extracts everything in `records`.
        pass

    with open(out_path, mode, encoding="utf-8") as f:
        ok, err = 0, 0
        for rec in tqdm(pending, desc=f"extracting ({args.corpus})"):
            text = f"{rec['title']}\n\n{rec['abstract']}"[:4000]
            try:
                result = call_gemini_with_retry(client, args.model, text, args.sleep)
                out_rec = {
                    "pmid": rec["pmid"],
                    "title": rec["title"],
                    "year": rec.get("year"),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    **result,
                }
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:
                print(f"  [error] pmid {rec['pmid']}: {exc}")
                err += 1
            f.flush()
            time.sleep(args.sleep)

    print(f"Done. Extracted OK: {ok}, errors: {err}. Output: {out_path}")


if __name__ == "__main__":
    main()
