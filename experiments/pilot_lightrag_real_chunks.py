"""Bounded real-corpus qualification on three predetermined common chunks.

Not a benchmark score: reduced index, no pilot question/answer/evidence used.
Use normal LightRAG insertion with a custom no-resplit chunking callback.
"""
import argparse
import asyncio
from importlib.metadata import version
import json
import hashlib
from pathlib import Path
import time

SOURCE = "Novel-47558"
START = 10
COUNT = 3
QUESTION = "Describe the people, places and events in the passages and how they are connected."
LLM = "Qwen/Qwen2.5-3B-Instruct"
EMBED = "BAAI/bge-large-en-v1.5"


def validate_query(result):
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"LightRAG retrieval did not succeed: {result}")
    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Missing structured retrieval data")
    counts = {}
    for name in ("entities", "relationships", "chunks"):
        rows = data.get(name)
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected {name} schema")
        counts[name] = len(rows)
    if min(counts.values()) == 0:
        raise RuntimeError(f"Smoke test requires retrieved graph entities, relations and chunks: {counts}")
    return counts


def single_chunk(tokenizer, content, split_by_character, split_by_character_only,
                 chunk_overlap_token_size, chunk_token_size):
    """Each input document is already one exact baseline chunk."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty shared chunk")
    return [{"tokens": len(tokenizer.encode(content)), "content": content,
             "chunk_order_index": 0}]


def check_storage(stored, expected, doc_id, file_path):
    if not isinstance(stored, dict):
        raise RuntimeError("missing stored chunk")
    if (stored.get("content") != expected or stored.get("full_doc_id") != doc_id
            or stored.get("file_path") != file_path):
        raise RuntimeError("shared passage content or provenance mismatch")


async def run(workdir, root):
    import torch
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from pilot_llm_generate import chat_inputs
    from pilot_retrieval import corpus_map, split_tokens
    from lightrag.utils import compute_mdhash_id

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU (T4) in Colab")
    if version("lightrag-hku") != "1.4.11":
        raise RuntimeError("Expected lightrag-hku==1.4.11")
    if workdir.exists():
        raise FileExistsError("Use a new smoke directory; prior graph is preserved")
    workdir.mkdir(parents=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    source_text = corpus_map(root)[("novel", SOURCE)]
    bge_tokenizer = AutoTokenizer.from_pretrained(EMBED)
    all_chunks = split_tokens(source_text, bge_tokenizer, size=256, overlap=32)
    if len(all_chunks) != 149 or len(set(all_chunks)) != 149:
        raise RuntimeError("Source inventory changed, expected 149 unique chunks")
    selected = all_chunks[START:START + COUNT]
    manifest = []
    for index, content in enumerate(selected, START):
        manifest.append({
            "chunk_index": index, "text": content,
            "doc_id": f"doc-{SOURCE}-{index:06d}",
            "chunk_id": compute_mdhash_id(content, prefix="chunk-"),
            "file_path": f"{SOURCE}/chunk-{index:06d}",
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    (workdir / "shared_chunk_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected common chunk indices:", [m["chunk_index"] for m in manifest], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LLM, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    encoder = SentenceTransformer(EMBED, device="cuda")
    calls = []
    lock = asyncio.Lock()

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # Keep graph extraction prompts intact; no RAG answer-system prompt here.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        async with lock:
            if len(calls) >= 18:
                raise RuntimeError("Smoke budget exhausted: maximum 18 LLM calls")
            tokens, mask = chat_inputs(tokenizer, messages, 8192)
            tokens, mask = tokens.to(model.device), mask.to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    input_ids=tokens, attention_mask=mask, max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                )
            new = out[0, tokens.shape[-1]:]
            answer = tokenizer.decode(new, skip_special_tokens=True).strip()
            record = {"call": len(calls) + 1, "input_tokens": int(tokens.shape[-1]),
                      "generated_tokens": int(new.numel()),
                      "elapsed_seconds": round(time.perf_counter() - started, 3),
                      "hit_token_cap": int(new.numel()) >= 1024}
            calls.append(record)
            print("LLM:", record, flush=True)
            if not answer or record["hit_token_cap"]:
                raise RuntimeError("Empty or potentially truncated graph extraction/keyword output")
            return answer

    async def embeddings(texts, **kwargs):
        async with lock:
            return encoder.encode(list(texts), batch_size=4,
                                  normalize_embeddings=True, show_progress_bar=False)

    rag = LightRAG(
        working_dir=str(workdir),
        llm_model_func=llm_func,
        llm_model_name=LLM,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=512,
                                     func=embeddings),
        chunking_func=single_chunk,
        chunk_token_size=256,
        chunk_overlap_token_size=32,
        llm_model_max_async=1,
        embedding_func_max_async=1,
        embedding_batch_num=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    timings = []
    started = time.perf_counter()
    try:
        for item in manifest:
            t0 = time.perf_counter()
            before = len(calls)
            await rag.ainsert(item["text"], ids=item["doc_id"],
                              file_paths=item["file_path"])
            doc = await rag.doc_status.get_by_id(item["doc_id"])
            if not isinstance(doc, dict) or doc.get("status") != "processed":
                raise RuntimeError(f"Document not processed: {item['doc_id']}: {doc}")
            stored = await rag.text_chunks.get_by_id(item["chunk_id"])
            check_storage(stored, item["text"], item["doc_id"], item["file_path"])
            row = {"chunk_index": item["chunk_index"],
                   "seconds": round(time.perf_counter() - t0, 3),
                   "llm_calls": len(calls) - before}
            timings.append(row)
            print("INDEXED:", row, flush=True)
            (workdir / "index_progress.json").write_text(
                json.dumps(timings, indent=2), encoding="utf-8")
        indexed_seconds = round(time.perf_counter() - started, 3)
        query_started = time.perf_counter()
        result = await rag.aquery_data(
            QUESTION, param=QueryParam(mode="hybrid", top_k=5, chunk_top_k=5,
                                       enable_rerank=False)
        )
        counts = validate_query(result)
        allowed = {m["text"] for m in manifest}
        for hit in result["data"]["chunks"]:
            if hit.get("content") not in allowed:
                raise RuntimeError("Retrieved passage differs from shared input")
        query_seconds = round(time.perf_counter() - query_started, 3)
    finally:
        await rag.finalize_storages()

    graphs = list(workdir.rglob("*.graphml"))
    if len(graphs) != 1:
        raise RuntimeError("Expected one persisted graph")
    graph = nx.read_graphml(graphs[0])
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        raise RuntimeError("No nontrivial graph")
    report = {
        "status": "REAL_CHUNKS_SMOKE_PASS",
        "scope": "three predetermined chunks; NOT a full-corpus comparison",
        "source": SOURCE, "indices": [m["chunk_index"] for m in manifest],
        "lightrag": version("lightrag-hku"), "transformers": version("transformers"),
        "llm": LLM, "embedding": EMBED,
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "query": QUESTION, "retrieved_counts": counts,
        "index_seconds": indexed_seconds, "query_seconds": query_seconds,
        "per_chunk": timings, "llm_calls": calls,
        "limitations": "Each common passage is a separate document in this smoke. No pilot answers or evidence were used. Graph semantics and multi-hop reasoning not yet validated.",
    }
    (workdir / "real_retrieval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "real_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.workdir, args.root))
