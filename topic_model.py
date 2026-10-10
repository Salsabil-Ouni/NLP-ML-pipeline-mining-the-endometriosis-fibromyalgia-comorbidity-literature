"""Step 3: topic modeling on a corpus, to map which themes already bridge
endometriosis and fibromyalgia (joint corpus) versus what's discussed in each
literature separately (endo/fibro corpora), for comparison.

Uses BERTopic (transformer embeddings + clustering) rather than classical LDA as the
primary method, since the joint corpus is small (order of 100s of papers) -- LDA
needs volume to find stable topics, BERTopic's embedding-based clustering handles
small corpora better. Run --method lda as a comparison baseline (methodologically
useful to report both and show where they agree/disagree, not just pick one).

Usage:
    python topic_model.py --corpus joint
    python topic_model.py --corpus joint --method lda
    python topic_model.py --corpus endo --min-topic-size 30   # larger corpus, larger min cluster
"""

import argparse
import json

import config
from extract_claims import load_jsonl


def run_bertopic(docs: list[str], min_topic_size: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import normalize
    from umap import UMAP

    # Precomputed TF-IDF+SVD embeddings instead of a sentence-transformers model:
    # this environment's torch install fails a Windows DLL init (c10.dll) even after
    # a clean reinstall, unrelated to this project's code, so a transformer-embedding
    # backend isn't reliably available here. BERTopic accepts any precomputed
    # embedding matrix via fit_transform(docs, embeddings=...), so the clustering
    # pipeline (UMAP + HDBSCAN + c-TF-IDF topic words) is unaffected.
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1, 2))
    X = tfidf.fit_transform(docs)
    n_components = min(300, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = normalize(svd.fit_transform(X))

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    umap_model = UMAP(n_neighbors=8, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=2, metric="euclidean",
                             cluster_selection_method="leaf", prediction_data=True)
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    topics_out = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue  # BERTopic's "outlier" bucket, not a real theme
        words = [w for w, _ in topic_model.get_topic(topic_id)]
        topics_out.append({
            "topic_id": int(topic_id),
            "size": int(row["Count"]),
            "top_words": words[:10],
        })

    return topics_out, [int(t) for t in topics]


def run_lda(docs: list[str], n_topics: int):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    doc_topic = lda.fit_transform(X)
    assignments = doc_topic.argmax(axis=1).tolist()

    topics_out = []
    for topic_id, topic_weights in enumerate(lda.components_):
        top_idx = topic_weights.argsort()[::-1][:10]
        words = [feature_names[i] for i in top_idx]
        size = sum(1 for a in assignments if a == topic_id)
        topics_out.append({"topic_id": topic_id, "size": size, "top_words": words})

    return topics_out, assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="joint")
    parser.add_argument("--method", choices=["bertopic", "lda"], default="bertopic")
    parser.add_argument("--min-topic-size", type=int, default=5, help="BERTopic: min docs per topic")
    parser.add_argument("--n-topics", type=int, default=5, help="LDA: number of topics")
    args = parser.parse_args()

    records = load_jsonl(config.RAW_DIR / f"{args.corpus}.jsonl")
    records = [r for r in records if r.get("abstract")]
    if len(records) < 10:
        raise SystemExit(f"Only {len(records)} records with abstracts -- too few for topic modeling. "
                          f"Fetch more with pubmed_fetch.py first.")

    docs = [f"{r['title']}. {r['abstract']}" for r in records]
    pmids = [r["pmid"] for r in records]

    print(f"Running {args.method} on {len(docs)} documents from corpus '{args.corpus}'...")

    if args.method == "bertopic":
        topics_out, assignments = run_bertopic(docs, args.min_topic_size)
    else:
        topics_out, assignments = run_lda(docs, args.n_topics)

    doc_assignments = [{"pmid": pmid, "topic_id": topic_id} for pmid, topic_id in zip(pmids, assignments)]

    out = {
        "corpus": args.corpus,
        "method": args.method,
        "n_documents": len(docs),
        "topics": topics_out,
        "document_assignments": doc_assignments,
    }

    out_path = config.TOPICS_DIR / f"{args.corpus}_{args.method}_topics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nFound {len(topics_out)} topics:")
    for t in sorted(topics_out, key=lambda x: -x["size"]):
        print(f"  Topic {t['topic_id']} (n={t['size']}): {', '.join(t['top_words'][:6])}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
