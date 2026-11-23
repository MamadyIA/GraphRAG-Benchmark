"""Fair pilot retrieval comparison: lexical TF-IDF vs BGE on IDENTICAL chunks.

Scores are NOT comparable across models. This script does not measure
retrieval quality automatically: it exports blinded-to-scoring review rows
for evidence adjudication.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_retrieval import (
    LexicalIndex, SemanticIndex, corpus_map, load_pilot, split_tokens,
)


METHODS = ("tfidf_shared", "bge_shared")


def compare(root, pilot, model, chunk_size=256, overlap=32, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("invalid chunk_size or overlap")
    sources = corpus_map(root)
    cache, output = {}, {method: [] for method in METHODS}
    summary = []

    for question in pilot:
        key = (question["domain"], question["source"])
        if key not in sources:
            raise ValueError(f"unknown source: {key}")
        if key not in cache:
            chunks = split_tokens(sources[key], model.tokenizer, chunk_size, overlap)
            if not chunks:
                raise ValueError(f"empty index: {key}")
            cache[key] = (LexicalIndex(chunks), SemanticIndex(chunks, model))
        lexical, semantic = cache[key]
        results = (lexical.search(question["question"], top_k),
                   semantic.search(question["question"], top_k))
        for method, hits in zip(METHODS, results):
            output[method].append({
                "id": question["id"],
                "domain": question["domain"],
                "source": question["source"],
                "question_type": question["question_type"],
                "question": question["question"],
                "ground_truth": question["answer"],
                "evidence": question["evidence"],
                "context": [hit["text"] for hit in hits],
                "retrieved_chunks": hits,
                "retrieval_mode": method,
                "generated_answer": None,
            })
        lexical_ids = {hit["chunk_index"] for hit in results[0]}
        semantic_ids = {hit["chunk_index"] for hit in results[1]}
        summary.append({
            "id": question["id"],
            "domain": question["domain"],
            "question_type": question["question_type"],
            "lexical_top1_chunk": results[0][0]["chunk_index"],
            "semantic_top1_chunk": results[1][0]["chunk_index"],
            "shared_top_k_chunks": len(lexical_ids & semantic_ids),
            "k": min(top_k, len(lexical.chunks)),
        })
    return output, summary, len(cache)


def write_outputs(outdir, output, summary, source_count, model_name,
                  chunk_size, overlap, top_k):
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for method in METHODS:
        path = outdir / f"pilot_{method}_retrieval.json"
        path.write_text(json.dumps(output[method], ensure_ascii=False, indent=2),
                        encoding="utf-8")
        files[method] = path
    report = {
        "model": model_name,
        "chunk_unit": "model tokenizer tokens",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "sources_indexed": source_count,
        "comparison": summary,
        "interpretation": (
            "shared_top_k_chunks counts passage overlap only. "
            "It is not evidence recall, factual accuracy, or answer quality. "
            "Similarity scores across retrieval modes are not comparable."
        ),
    }
    report_path = outdir / "pilot_shared_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    review_path = outdir / "pilot_shared_manual_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "question_id", "domain", "question_type", "question", "reference_evidence",
            "method", "rank", "chunk_index", "score", "retrieved_passage",
            "supports_evidence_yes_no_partial", "reviewer_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, items in output.items():
            for item in items:
                for passage in item["retrieved_chunks"]:
                    writer.writerow({
                        "question_id": item["id"],
                        "domain": item["domain"],
                        "question_type": item["question_type"],
                        "question": item["question"],
                        "reference_evidence": item["evidence"],
                        "method": method,
                        "rank": passage["rank"],
                        "chunk_index": passage["chunk_index"],
                        "score": passage["score"],
                        "retrieved_passage": passage["text"],
                        "supports_evidence_yes_no_partial": "",
                        "reviewer_notes": "",
                    })
    return files, report_path, review_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; choose --device cpu or enable Colab GPU.")
    model = SentenceTransformer(args.model, device=args.device)
    pilot = load_pilot(args.pilot)
    results, summary, sources = compare(
        args.root, pilot, model, args.chunk_size, args.overlap, args.top_k
    )
    files, report, review = write_outputs(
        args.output_dir, results, summary, sources, args.model,
        args.chunk_size, args.overlap, args.top_k
    )
    print("Pilot questions:", len(summary))
    print("Sources:", sources, "; Common chunking:", args.chunk_size, "tokens")
    print("Shared top-k passages (not an accuracy metric):",
          sum(record["shared_top_k_chunks"] for record in summary))
    for method, path in files.items():
        print(method, path)
    print("Comparison:", report)
    print("Human-evidence review sheet:", review)


if __name__ == "__main__":
    main()
