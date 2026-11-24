"""Bounded LightRAG smoke test on synthetic text, not the benchmark pilot.

The local LLM is used for graph extraction and query keyword extraction.
No reference answers, benchmark text or paid API are used here.
"""
import argparse
import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import time

TEXT = (
    "Mira founded the Cedar Research Lab in Riverton. "
    "The Cedar Research Lab develops water filtration systems. "
    "Jonas works at Cedar Research Lab and leads Project ClearWater. "
    "Project ClearWater tests filters at the River Stone field station. "
    "The River Stone field station belongs to the Cedar Research Lab."
)
QUESTION = "Which organization employs Jonas and what project does he lead?"
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


async def run(workdir):
    import torch
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from pilot_llm_generate import chat_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU (T4) in Colab")
    if version("lightrag-hku") != "1.4.11":
        raise RuntimeError("Expected lightrag-hku==1.4.11")
    if workdir.exists():
        raise FileExistsError("Use a new smoke directory; prior graph is preserved")
    workdir.mkdir(parents=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

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
            if len(calls) >= 12:
                raise RuntimeError("Smoke budget exhausted: maximum 12 LLM calls")
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
        chunk_token_size=256,
        chunk_overlap_token_size=32,
        llm_model_max_async=1,
        embedding_func_max_async=1,
        embedding_batch_num=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    started = time.perf_counter()
    try:
        await rag.ainsert(TEXT)
        indexed_seconds = round(time.perf_counter() - started, 3)
        query_started = time.perf_counter()
        result = await rag.aquery_data(
            QUESTION, param=QueryParam(mode="hybrid", top_k=5, chunk_top_k=5,
                                       enable_rerank=False)
        )
        counts = validate_query(result)
        query_seconds = round(time.perf_counter() - query_started, 3)
    finally:
        await rag.finalize_storages()

    graphs = list(workdir.rglob("*.graphml"))
    if len(graphs) != 1:
        raise RuntimeError(f"Expected exactly one persisted GraphML graph, got {len(graphs)}")
    graph = nx.read_graphml(graphs[0])
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        raise RuntimeError("No nontrivial knowledge graph persisted")
    report = {
        "status": "SMOKE_PASS", "scope": "synthetic; not benchmark",
        "lightrag": version("lightrag-hku"), "transformers": version("transformers"),
        "llm": LLM, "embedding": EMBED, "query": QUESTION, "mode": "hybrid",
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "retrieved_counts": counts, "index_seconds": indexed_seconds,
        "query_seconds": query_seconds, "llm_calls": calls,
        "limitations": "Technical smoke only; no correctness or benchmark scoring. Native LightRAG tokenization and unprefixed BGE embeddings differ from the controlled baseline; qualify shared chunking before full experiment.",
    }
    (workdir / "smoke_retrieval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (workdir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.workdir))
