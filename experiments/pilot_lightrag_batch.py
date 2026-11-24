"""One source, one bounded batch per invocation; verified ZIP checkpoints.
Never run concurrent writers. A failed batch is not published.
"""
import argparse
import asyncio
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED
from pilot_lightrag_real_chunks import single_chunk, check_storage, LLM, EMBED

SOURCES = {"Medical": ("medical", 1030), "Novel-10146": ("novel", 231),
           "Novel-30752": ("novel", 174), "Novel-47558": ("novel", 149),
           "Novel-5956": ("novel", 271)}


# Internal graph extraction budget, distinct from the 384-token final answer budget.
GENERATION_CAPS = (2048, 3072)
COMPLETION_MARKER = "<|COMPLETE|>"
CHECKPOINT_VERSION = "v2"


def classify_generation(answer, generated_tokens, cap, require_marker, eos_reached):
    """Never publish an incomplete extraction or an unterminated summary."""
    answer = answer.strip()
    if not answer:
        return "empty"
    if require_marker and answer.endswith(COMPLETION_MARKER):
        return "complete_marker"
    if generated_tokens >= cap and not eos_reached:
        return "token_cap"
    if require_marker:
        return "missing_completion_marker"
    if eos_reached:
        return "complete_eos"
    return "unexpected_stop"


def bounded_generation(generate_once, require_marker):
    """Retry only a length-limited response; all other anomalies fail closed."""
    for cap in GENERATION_CAPS:
        answer, n_tokens, eos_reached = generate_once(cap, require_marker)
        reason = classify_generation(answer, n_tokens, cap, require_marker, eos_reached)
        if reason in ("complete_marker", "complete_eos"):
            return answer.strip()
        if reason == "token_cap" and cap != GENERATION_CAPS[-1]:
            print(f"RETRY_LENGTH: {cap} -> {GENERATION_CAPS[-1]} tokens", flush=True)
            continue
        raise RuntimeError(f"Incomplete graph LLM output: {reason}; cap={cap}; "
                           f"tokens={n_tokens}; tail={answer[-300:]!r}")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def check_state(state, config, count):
    if not isinstance(state, dict) or state.get("config") != config:
        raise ValueError("Checkpoint configuration differs; do not mix experiments.")
    rows = state.get("progress")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Checkpoint progress length mismatch")
    if [r["chunk_index"] for r in rows] != list(range(count)):
        raise ValueError("Checkpoint indices are not a contiguous prefix")
    if not 0 < count <= config["total_chunks"]:
        raise ValueError("Invalid checkpoint count")


def restore_latest(folder, workdir, config):
    snapshots = sorted(folder.glob("index_*.zip"))
    if not snapshots:
        return []
    latest = snapshots[-1]
    count = int(latest.stem.split("_")[-1])
    with ZipFile(latest) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "batch_state.json" not in names:
            raise ValueError("Invalid checkpoint archive members")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe checkpoint archive path")
        if z.testzip() is not None:
            raise ValueError("Corrupted checkpoint ZIP")
        state = json.loads(z.read("batch_state.json"))
        check_state(state, config, count)
        z.extractall(workdir)
    print("RESTORED:", latest, count, "chunks", flush=True)
    return state["progress"]


def publish(folder, workdir, config, progress):
    count = len(progress)
    state = {"config": config, "progress": progress}
    check_state(state, config, count)
    (workdir / "batch_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    target = folder / f"index_{count:06d}.zip"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    fd, tmp_name = tempfile.mkstemp(prefix=".index_", suffix=".partial", dir=folder)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
            for path in sorted(workdir.rglob("*")):
                if path.is_file():
                    z.write(path, arcname=path.relative_to(workdir).as_posix())
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError("Snapshot CRC verification failed")
            check_state(json.loads(z.read("batch_state.json")), config, count)
        if target.exists():
            raise FileExistsError("Concurrent checkpoint writer detected")
        os.replace(tmp, target)
        with ZipFile(target) as z:
            if z.testzip() is not None:
                raise RuntimeError("Published archive failed CRC verification")
        print("CHECKPOINT:", target, flush=True)
    finally:
        tmp.unlink(missing_ok=True)
    return target


async def run(workdir, root, folder, source, batch_size):
    import torch
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, compute_mdhash_id
    from lightrag.prompt import PROMPTS
    from pilot_llm_generate import chat_inputs
    from pilot_retrieval import corpus_map, split_tokens

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU in Colab")
    if version("lightrag-hku") != "1.4.11":
        raise RuntimeError("Expected lightrag-hku==1.4.11")
    if PROMPTS["DEFAULT_COMPLETION_DELIMITER"] != COMPLETION_MARKER:
        raise RuntimeError("LightRAG completion delimiter changed")
    workdir.mkdir(parents=True)
    domain, expected_count = SOURCES[source]
    source_text = corpus_map(root)[(domain, source)]
    bge_tokenizer = AutoTokenizer.from_pretrained(EMBED)
    all_chunks = split_tokens(source_text, bge_tokenizer, size=256, overlap=32)
    if len(all_chunks) != expected_count or len(set(all_chunks)) != expected_count:
        raise RuntimeError("Source inventory changed")
    manifest = []
    for index, content in enumerate(all_chunks):
        manifest.append({
            "chunk_index": index, "text": content,
            "doc_id": f"doc-{source}-{index:06d}",
            "chunk_id": compute_mdhash_id(content, prefix="chunk-"),
            "file_path": f"{source}/chunk-{index:06d}",
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    config = {"protocol": "shared-chunks-batch-v2", "source": source,
              "domain": domain, "total_chunks": expected_count,
              "manifest_sha256": digest(manifest), "llm": LLM, "embedding": EMBED,
              "chunk_size": 256, "overlap": 32, "max_input_tokens": 8192,
              "generation_caps": list(GENERATION_CAPS), "completion_marker": COMPLETION_MARKER,
              "require_extract_marker": True, "stop_on_marker": True,
              "gleaning": 0, "dtype": "float16",
              "versions": {name: version(name) for name in
                  ("lightrag-hku", "transformers", "torch", "sentence-transformers")}}
    folder.mkdir(parents=True, exist_ok=True)
    progress = restore_latest(folder, workdir, config)
    if len(progress) == expected_count:
        print("SOURCE_COMPLETE:", source, len(progress), "/", expected_count, flush=True)
        return
    selected = manifest[len(progress):len(progress) + batch_size]
    print("BATCH:", source, [m["chunk_index"] for m in selected], flush=True)
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
            tokens, mask = chat_inputs(tokenizer, messages, 8192)
            tokens, mask = tokens.to(model.device), mask.to(model.device)
            # The upstream extraction system prompt requires the literal marker.
            require_marker = COMPLETION_MARKER in (system_prompt or "")
            eos = model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

            def generate_once(cap, require_marker):
                if len(calls) >= max(30, 8 * batch_size):
                    raise RuntimeError("Batch call budget exhausted; previous checkpoint preserved")
                params = {}
                if require_marker:
                    params.update(stop_strings=[COMPLETION_MARKER], tokenizer=tokenizer)
                started = time.perf_counter()
                with torch.inference_mode():
                    out = model.generate(
                        input_ids=tokens, attention_mask=mask, max_new_tokens=cap,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id,
                        **params,
                    )
                new = out[0, tokens.shape[-1]:]
                answer = tokenizer.decode(new, skip_special_tokens=True).strip()
                n_tokens = int(new.numel())
                eos_reached = n_tokens > 0 and int(new[-1].item()) in eos_ids
                reason = classify_generation(answer, n_tokens, cap, require_marker, eos_reached)
                record = {"call": len(calls) + 1, "input_tokens": int(tokens.shape[-1]),
                          "generated_tokens": n_tokens, "max_new_tokens": cap,
                          "elapsed_seconds": round(time.perf_counter() - started, 3),
                          "hit_token_cap": n_tokens >= cap, "eos_reached": eos_reached,
                          "requires_completion_marker": require_marker, "finish_reason": reason}
                calls.append(record)
                print("LLM:", record, flush=True)
                return answer, n_tokens, eos_reached

            return bounded_generation(generate_once, require_marker)

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
        default_llm_timeout=360,
        embedding_func_max_async=1,
        embedding_batch_num=4,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    async def verify(item):
        doc = await rag.doc_status.get_by_id(item["doc_id"])
        if not isinstance(doc, dict) or doc.get("status") != "processed":
            raise RuntimeError(f"Document not processed: {item['doc_id']}: {doc}")
        stored = await rag.text_chunks.get_by_id(item["chunk_id"])
        check_storage(stored, item["text"], item["doc_id"], item["file_path"])

    try:
        for item in manifest[:len(progress)]:
            await verify(item)
        for item in selected:
            t0 = time.perf_counter()
            before = len(calls)
            await rag.ainsert(item["text"], ids=item["doc_id"], file_paths=item["file_path"])
            await verify(item)
            row = {"chunk_index": item["chunk_index"],
                   "seconds": round(time.perf_counter() - t0, 3),
                   "llm_calls": len(calls) - before}
            progress.append(row)
            print("INDEXED (not yet committed):", row, flush=True)
        for item in manifest[:len(progress)]:
            await verify(item)
    finally:
        await rag.finalize_storages()

    graphs = list(workdir.rglob("*.graphml"))
    if len(graphs) != 1:
        raise RuntimeError("Expected one persisted graph")
    graph = nx.read_graphml(graphs[0])
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        raise RuntimeError("No nontrivial graph")
    (workdir / "batch_report.json").write_text(json.dumps({
        "source": source, "indexed": len(progress), "total": expected_count,
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "batch_llm_calls": calls, "batch_seconds": sum(r["seconds"] for r in progress[-len(selected):]),
        "status": "SOURCE_COMPLETE" if len(progress) == expected_count else "BATCH_COMPLETE",
        "limitations": "Verified document status and exact chunk provenance, not semantic correctness."
    }, indent=2), encoding="utf-8")
    publish(folder, workdir, config, progress)
    print((workdir / "batch_report.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 25:
        parser.error("batch-size must be 1..25")
    with tempfile.TemporaryDirectory(prefix="lightrag_batch_") as scratch:
        asyncio.run(run(Path(scratch) / "graph", args.root,
                        args.checkpoint_dir / (args.source + "-" + CHECKPOINT_VERSION),
                        args.source, args.batch_size))
