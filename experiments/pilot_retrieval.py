"""Pilot retrieval-only baseline for GraphRAG-Bench.

No generation or scoring is performed. Uses only corpus contexts for retrieval:
question answers and evidence never enter the index.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def words(text):
    return text.split()


def split_words(text, size=256, overlap=32):
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must exceed overlap and both must be nonnegative")
    tokens = words(text)
    step = size - overlap
    return [" ".join(tokens[i:i + size]) for i in range(0, len(tokens), step)
            if tokens[i:i + size]]


def split_tokens(text, tokenizer, size=256, overlap=32):
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must exceed overlap")
    # Tokenizing an entire novel in one call can exceed a transformer's
    # maximum input length and cause huge intermediate allocations.
    # Tokenize short word groups, then create globally overlapping token chunks.
    token_ids = []
    units = text.split()
    for start in range(0, len(units), 128):
        fragment = " ".join(units[start:start + 128])
        token_ids.extend(tokenizer.encode(fragment, add_special_tokens=False))
    step = size - overlap
    return [tokenizer.decode(token_ids[i:i + size], skip_special_tokens=True)
            for i in range(0, len(token_ids), step) if token_ids[i:i + size]]


def corpus_map(root):
    data = {}
    for domain in ("medical", "novel"):
        with (root / "Datasets" / "Corpus" / f"{domain}.json").open(encoding="utf-8") as handle:
            corpus = json.load(handle)
        documents = [corpus] if domain == "medical" else corpus
        if not isinstance(documents, list):
            raise ValueError(f"invalid {domain} corpus")
        for document in documents:
            key = (domain, document["corpus_name"])
            if key in data:
                raise ValueError(f"duplicate corpus source: {key}")
            if not isinstance(document["context"], str) or not document["context"].strip():
                raise ValueError(f"empty corpus context: {key}")
            data[key] = document["context"]
    return data


def load_pilot(path):
    with path.open(encoding="utf-8") as handle:
        pilot = json.load(handle)
    if not isinstance(pilot, list) or not pilot:
        raise ValueError("pilot must be a nonempty list")
    identifiers = set()
    for item in pilot:
        if not isinstance(item, dict):
            raise ValueError("pilot row must be an object")
        for field in ("id", "domain", "source", "question", "question_type", "answer", "evidence"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"invalid or missing pilot field: {field}")
        if item["domain"] not in ("medical", "novel"):
            raise ValueError("unknown domain")
        if item["id"] in identifiers:
            raise ValueError(f"duplicate pilot id: {item['id']}")
        identifiers.add(item["id"])
    return pilot


def bag(text):
    return Counter(WORD_PATTERN.findall(text.casefold()))


class LexicalIndex:
    """TF-IDF cosine retrieval for a dependency-free smoke test, NOT paper's RAG."""
    def __init__(self, chunks):
        self.chunks = chunks
        counts = [bag(text) for text in chunks]
        document_frequency = Counter()
        for count in counts:
            document_frequency.update(count.keys())
        n = len(chunks)
        self.idf = {term: math.log((n + 1) / (df + 1)) + 1
                    for term, df in document_frequency.items()}
        self.vectors = [self.vector(count) for count in counts]

    def vector(self, counts):
        weighted = {term: (1 + math.log(freq)) * self.idf[term]
                    for term, freq in counts.items() if term in self.idf and freq > 0}
        magnitude = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / magnitude for term, value in weighted.items()} if magnitude else {}

    def search(self, query, top_k):
        vector = self.vector(bag(query))
        scores = [
            sum(weight * doc.get(term, 0.0) for term, weight in vector.items())
            for doc in self.vectors
        ]
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [{"rank": rank + 1, "score": round(scores[i], 8),
                 "text": self.chunks[i], "chunk_index": i}
                for rank, i in enumerate(ranking)]


class SemanticIndex:
    """BGE embedding cosine ranking; model loads only in semantic mode."""
    def __init__(self, chunks, model, batch_size=8):
        self.chunks = chunks
        self.model = model
        self.vectors = model.encode(
            chunks, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False
        )

    def search(self, query, top_k):
        prompt = "Represent this sentence for searching relevant passages: " + query
        query_vector = self.model.encode(
            [prompt], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.vectors @ query_vector
        ranking = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:top_k]
        return [{"rank": rank + 1, "score": round(float(scores[i]), 8),
                 "text": self.chunks[i], "chunk_index": i}
                for rank, i in enumerate(ranking)]


def retrieve(root, pilot, mode="lexical", chunk_size=256, overlap=32, top_k=5,
             model_name="BAAI/bge-large-en-v1.5"):
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if mode not in ("lexical", "semantic"):
        raise ValueError("mode must be lexical or semantic")
    corpora = corpus_map(root)
    model = None
    if mode == "semantic":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "semantic mode needs sentence-transformers in a separate environment"
            ) from exc
        model = SentenceTransformer(model_name)

    indices, results = {}, []
    for item in pilot:
        key = (item["domain"], item["source"])
        if key not in corpora:
            raise ValueError(f"pilot source does not exist: {key}")
        if key not in indices:
            context = corpora[key]
            chunks = (split_tokens(context, model.tokenizer, chunk_size, overlap)
                      if mode == "semantic" else split_words(context, chunk_size, overlap))
            if not chunks:
                raise ValueError(f"no chunks for {key}")
            indices[key] = (SemanticIndex(chunks, model) if mode == "semantic"
                            else LexicalIndex(chunks))
        selected = indices[key].search(item["question"], top_k)
        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "source": item["source"],
            "question_type": item["question_type"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "evidence": item["evidence"],
            "context": [candidate["text"] for candidate in selected],
            "retrieved_chunks": selected,
            "retrieval_mode": mode,
            "generated_answer": None,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["lexical", "semantic"], default="lexical")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    pilot = load_pilot(args.pilot)
    results = retrieve(
        args.root, pilot, args.mode, args.chunk_size, args.overlap,
        args.top_k, args.model
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Retrieval finished: {len(results)} questions; mode={args.mode}")
    print(f"Unique sources indexed: {len({(r['domain'], r['source']) for r in results})}")
    print(f"Output: {args.output}")
    print("No answer generation or benchmark scoring performed.")


if __name__ == "__main__":
    main()
