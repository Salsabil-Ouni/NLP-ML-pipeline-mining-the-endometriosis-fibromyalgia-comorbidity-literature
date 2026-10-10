# endo-fibro-lit-mining

NLP/ML pipeline mining the endometriosis-fibromyalgia comorbidity literature: corpus
construction with a built-in false-positive audit, LLM-based structured claim
extraction (validated against independent human annotation), comparative topic
modeling, and a retrieval-augmented method for surfacing papers in the separate
endometriosis and fibromyalgia literatures that describe the same mechanism without
citing each other.

Built entirely from free tools and public PubMed data (no paid APIs, no
human-subjects data, no ethics approval required).

Companion code/results article to the mini-review "Mining the comorbidity: Natural
Language Processing and Machine Learning Approaches to the Endometriosis-Fibromyalgia
Link" (Higher Institute of Biotechnology of Sidi Thabet, University of Manouba).

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a free Gemini API key (no card required):
https://aistudio.google.com/apikey

## Pipeline

Run in this order. Each script is independently re-runnable; outputs are cached/
resumable so an interrupted run can just be re-started.

1. **`pubmed_fetch.py`** -- fetch the three corpora from PubMed (free, no key beyond
   an email address, which NCBI requires on every request).
   ```
   python pubmed_fetch.py --email you@example.com --corpus joint
   python pubmed_fetch.py --email you@example.com --corpus endo
   python pubmed_fetch.py --email you@example.com --corpus fibro
   ```
2. **`audit_corpus.py`** -- measure and flag records that matched a query via
   PubMed's automatic term-mapping but don't literally contain the required disease
   term(s). Run on each corpus, then manually exclude the flagged PMIDs before
   downstream analysis (a short filter script, not included here since it's a
   one-line `jq`/pandas operation on the flagged-list output).
   ```
   python audit_corpus.py --corpus joint
   python audit_corpus.py --corpus endo
   python audit_corpus.py --corpus fibro
   ```
3. **`extract_claims.py`** -- LLM structured extraction (mechanism, study design,
   sample size, comorbidity focus, fibromyalgia-measurement type, stance) on the
   joint corpus.
   ```
   python extract_claims.py --corpus joint
   ```
4. **`sample_for_validation.py`** / **`validate_extraction.py`** -- build a random
   sample for manual annotation, then compare it against the LLM extraction
   (raw agreement + Cohen's kappa per field).
   ```
   python sample_for_validation.py --corpus joint --n 15
   # fill in data/validation/joint_annotation_template.csv by hand, save as
   # joint_annotation_template_completed.csv, then:
   python validate_extraction.py --corpus joint
   ```
5. **`topic_model.py`** -- BERTopic (primary) and LDA (baseline) on any corpus.
   ```
   python topic_model.py --corpus joint --method bertopic
   python topic_model.py --corpus joint --method lda
   ```
6. **`rag_discovery.py`** -- the flagship step: find papers in the separate
   endometriosis-only and fibromyalgia-only literatures that describe the same
   mechanism without citing each other or appearing in the joint corpus.
   ```
   python rag_discovery.py
   ```

`scripts/build_figures.py` and `scripts/build_article_docx.py` regenerate the
article draft and its figures from whatever is currently in `data/`; they are not
part of the core pipeline and exist only to produce the write-up.

## Notes on reproducibility

- **Embeddings are TF-IDF + truncated SVD (scikit-learn), not transformer sentence
  embeddings.** This was a deliberate substitution after `sentence-transformers`
  (which pulls in PyTorch) failed to load with a Windows DLL initialization error
  in the original development environment, unrelated to this code. If
  `sentence-transformers` works reliably in your environment, swapping it back in
  is straightforward, but results reported in the article were produced with the
  TF-IDF/SVD backend.
- **LLM extraction is not perfectly deterministic.** Repeated runs with an
  identical prompt showed several-percentage-point fluctuation on some fields even
  with structured JSON output; treat any single validation run as an estimate.
- **PubMed's search API silently admits term-mapping false positives.** Always run
  `audit_corpus.py` and exclude flagged records before reporting corpus sizes or
  trusting that a given record actually mentions the disease term it was searched
  for.


