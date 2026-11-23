"""Read-only GraphRAG-Bench JSON audit and reproducible 12-question pilot."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

TYPES = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation")
QUOTAS = {domain: dict(zip(TYPES, (2, 2, 1, 1))) for domain in ("medical", "novel")}


def read(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def corpus_sources(domain, data):
    documents = [data] if domain == "medical" else data
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{domain}: invalid corpus layout")
    names = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError(f"{domain}: corpus document must be an object")
        name, context = document.get("corpus_name"), document.get("context")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{domain}: empty corpus_name")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"{domain}: missing context for {name}")
        if name in names:
            raise ValueError(f"{domain}: duplicate corpus_name {name}")
        names.add(name)
    return names


def audit(root, seed=42):
    rows, counts, domains = [], {}, {}
    for domain in ("medical", "novel"):
        base = root / "Datasets"
        domains[domain] = corpus_sources(domain, read(base / "Corpus" / f"{domain}.json"))
        questions = read(base / "Questions" / f"{domain}_questions.json")
        if not isinstance(questions, list):
            raise ValueError(f"{domain}: questions must be a list")
        counts[domain] = {
            "questions": len(questions),
            "corpus_sources": len(domains[domain]),
            "by_type": dict(sorted(Counter(
                row.get("question_type", "unknown") if isinstance(row, dict) else "invalid"
                for row in questions
            ).items())),
        }
        rows.extend((domain, i, row) for i, row in enumerate(questions))

    issues = defaultdict(set)
    by_id, by_question = defaultdict(list), defaultdict(list)
    lookup = {}
    for domain, index, row in rows:
        key = (domain, index)
        lookup[key] = row
        if not isinstance(row, dict):
            issues[key].add("not_an_object")
            continue
        for field in ("id", "source", "question", "answer", "evidence"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                issues[key].add("invalid_" + field)
        if row.get("question_type") not in TYPES:
            issues[key].add("unknown_question_type")
        if row.get("source") not in domains[domain]:
            issues[key].add("source_not_in_corpus")
        identifier = row.get("id")
        if isinstance(identifier, str) and identifier.strip():
            by_id[identifier].append(key)
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            by_question[(domain, " ".join(question.casefold().split()))].append(key)

    duplicates = []
    for identifier, keys in sorted(by_id.items()):
        if len(keys) > 1:
            duplicates.append({"id": identifier, "locations": [
                {"domain": d, "index": i} for d, i in keys
            ]})
            for key in keys:
                issues[key].add("duplicate_id")

    conflicting_references = []
    for (domain, question), keys in sorted(by_question.items()):
        answers = {lookup[k]["answer"].strip().casefold() for k in keys
                   if isinstance(lookup[k].get("answer"), str)}
        if len(answers) > 1:
            conflicting_references.append({
                "domain": domain, "question_excerpt": question[:160],
                "indices": [i for _, i in keys]
            })
            for key in keys:
                issues[key].add("conflicting_reference")

    eligible = [(d, i, row) for d, i, row in rows if (d, i) not in issues]
    details = [{"domain": d, "index": i, "id": row.get("id") if isinstance(row, dict) else None,
                "reasons": sorted(issues[d, i])}
               for d, i, row in rows if (d, i) in issues]
    report = {
        "seed": seed, "counts": counts, "rows_checked": len(rows),
        "eligible_rows": len(eligible), "excluded_rows": len(details),
        "issue_counts": dict(sorted(Counter(
            reason for item in details for reason in item["reasons"]
        ).items())),
        "duplicate_ids": duplicates,
        "conflicting_references": conflicting_references,
        "issues": details,
        "limitation": "Structural checks only; factual and evidence correctness need human review.",
    }

    rng = random.Random(seed)
    pilot = []
    for domain, quotas in QUOTAS.items():
        for kind, amount in quotas.items():
            pool = sorted((i, row) for d, i, row in eligible
                          if d == domain and row["question_type"] == kind)
            if len(pool) < amount:
                raise ValueError(f"Insufficient valid records: {domain}/{kind} ({len(pool)}/{amount})")
            for index, row in rng.sample(pool, amount):
                pilot.append({"domain": domain, "original_index": index, **row})
    pilot.sort(key=lambda r: (r["domain"], TYPES.index(r["question_type"]), r["id"]))
    report["pilot_size"] = len(pilot)
    report["pilot_distribution"] = {
        domain: {kind: sum(x["domain"] == domain and x["question_type"] == kind
                           for x in pilot) for kind in TYPES}
        for domain in QUOTAS
    }
    return report, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report, pilot = audit(root, args.seed)
    destination = args.output_dir or root.parent / "graphrag_audit_results"
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in (("data_quality_report.json", report),
                          ("pilot_12_questions.json", pilot)):
        with (destination / name).open("w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)
    print(f"Rows checked: {report['rows_checked']}")
    print(f"Excluded rows: {report['excluded_rows']}")
    print(f"Issue counts: {report['issue_counts']}")
    print(f"Pilot questions: {len(pilot)}")
    print(f"Reports: {destination}")
    print("Original datasets unchanged.")


if __name__ == "__main__":
    main()
