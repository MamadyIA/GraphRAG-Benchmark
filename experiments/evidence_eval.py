"""Provisional and human-validated evidence annotations for GraphRAG pilot.

Strict facet coverage counts ONLY evidence facets with an explicit 'yes' match
in a retrieved passage; partial matches are reported but not scored as success.
If every annotation is not human validated, outputs are PROVISIONAL and
MUST NOT be published as ground-truth benchmark metrics.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

METHODS = ("tfidf_shared", "bge_shared")
REVIEW_VALUES = ("ai_prelabel", "human_validated")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_facet_ids(value):
    return set(int(item.strip()) for item in value.split(";") if item.strip())


def qualify(pilot, rubric, rows, require_human=False):
    if len({p['id'] for p in pilot}) != len(pilot):
        raise ValueError("pilot contains duplicate IDs")
    questions = {p['id']: p for p in pilot}
    rubrics = {r['id']: r for r in rubric}
    if set(questions) != set(rubrics):
        raise ValueError("rubric does not match pilot IDs")
    for qid, item in rubrics.items():
        official = [f.strip() for f in questions[qid]['evidence'].split(';') if f.strip()]
        provided = item['facets']
        if len(provided) != len(official) or any(f['text'] != expected or f['index'] != index
                 for index, (f, expected) in enumerate(zip(provided, official), start=1)):
            raise ValueError(f"rubric facets differ from the official evidence: {qid}")
        if any(f['scope'] not in ('source', 'question_given', 'interpretive') for f in provided):
            raise ValueError(f"unknown evidence scope for {qid}")
        if not any(f['scope'] == 'source' for f in provided):
            raise ValueError(f"no scorable facet for {qid}")

    by_question_method = defaultdict(list)
    identities = set()
    for row in rows:
        qid, method = row['question_id'], row['method']
        if qid not in questions or method not in METHODS:
            raise ValueError(f"unknown question or method: {qid}/{method}")
        rank = int(row['rank'])
        if rank < 1 or rank > 5:
            raise ValueError("ranks must be within 1..5")
        identity = (qid, method, rank)
        if identity in identities:
            raise ValueError(f"duplicate annotation row: {identity}")
        identities.add(identity)
        yes = parse_facet_ids(row.get('facet_ids_yes', ''))
        partial = parse_facet_ids(row.get('facet_ids_partial', ''))
        if yes & partial:
            raise ValueError(f"same facet both yes and partial: {identity}")
        allowed = {f['index'] for f in rubrics[qid]['facets'] if f['scope'] == 'source'}
        if not (yes | partial) <= allowed:
            raise ValueError(f"annotations refer to non-source facets: {identity}")
        label = row['candidate_label']
        if label not in ('yes', 'partial', 'no'):
            raise ValueError(f"unrecognized label: {label}")
        if label != ('yes' if yes else ('partial' if partial else 'no')):
            raise ValueError(f"label/facet mismatch: {identity}")
        if row['review_status'] not in REVIEW_VALUES:
            raise ValueError(f"invalid review status: {identity}")
        if require_human and row['review_status'] != 'human_validated':
            raise ValueError("human-validated annotations required for finalized metrics")
        if not row.get('retrieved_passage', '').strip():
            raise ValueError(f"empty retrieved passage: {identity}")
        by_question_method[(qid, method)].append({**row,
            'rank': rank, 'facet_ids_yes_set': yes, 'facet_ids_partial_set': partial})

    total_expected = len(pilot) * len(METHODS) * 5
    if len(rows) != total_expected:
        raise ValueError(f"expected {total_expected} annotation rows, found {len(rows)}")

    details = []
    for qid, question in questions.items():
        scope = {f['index']: f['scope'] for f in rubrics[qid]['facets']}
        scorable = {index for index, s in scope.items() if s == 'source'}
        for method in METHODS:
            hits = sorted(by_question_method[(qid, method)], key=lambda a: a['rank'])
            if [x['rank'] for x in hits] != [1,2,3,4,5]:
                raise ValueError(f"missing/non-consecutive ranks: {qid}/{method}")
            found = set().union(*(x['facet_ids_yes_set'] for x in hits))
            partial = set().union(*(x['facet_ids_partial_set'] for x in hits)) - found
            yes_hits = [x for x in hits if x['facet_ids_yes_set']]
            details.append({
                'id': qid, 'domain': question['domain'],
                'question_type': question['question_type'], 'method': method,
                'source_facet_count': len(scorable),
                'source_facets_found': len(found),
                'facets_found': sorted(found),
                'facets_partial_only': sorted(partial),
                'facets_missing': sorted(scorable - found),
                'any_strict_evidence_hit_at_5': bool(yes_hits),
                'complete_strict_facet_coverage_at_5': found == scorable,
                'strict_reciprocal_rank': 1 / min(x['rank'] for x in yes_hits) if yes_hits else 0.0,
                'strict_relevant_chunks_at_5': len(yes_hits),
                'partial_only_chunks_at_5': sum(not x['facet_ids_yes_set'] and bool(x['facet_ids_partial_set']) for x in hits),
            })

    def summarise(items):
        count = len(items)
        denominator = sum(x['source_facet_count'] for x in items)
        return {
            'questions': count,
            'strict_any_evidence_hit_at_5': sum(x['any_strict_evidence_hit_at_5'] for x in items) / count,
            'strict_complete_facet_coverage_at_5': sum(x['complete_strict_facet_coverage_at_5'] for x in items) / count,
            'strict_facet_recall_at_5': sum(x['source_facets_found'] for x in items) / denominator,
            'strict_relevant_chunk_precision_at_5': sum(x['strict_relevant_chunks_at_5'] for x in items) / (5 * count),
            'mean_reciprocal_rank_at_5': statistics.mean(x['strict_reciprocal_rank'] for x in items),
            'total_source_facets': denominator,
            'total_source_facets_found': sum(x['source_facets_found'] for x in items),
            'strict_relevant_chunks': sum(x['strict_relevant_chunks_at_5'] for x in items),
            'partial_only_chunks': sum(x['partial_only_chunks_at_5'] for x in items),
        }

    provisional = any(row['review_status'] != 'human_validated' for row in rows)
    return {
        'status': 'PROVISIONAL_AI_ANNOTATION_NOT_HUMAN_VALIDATED' if provisional else 'HUMAN_VALIDATED_ANNOTATIONS',
        'metric_definitions': {
            'strict_any_evidence_hit_at_5': 'Fraction of questions with at least one passage explicitly supporting at least one source facet.',
            'strict_complete_facet_coverage_at_5': 'Fraction of questions with all source facets explicitly supported by Top-5 union.',
            'strict_facet_recall_at_5': 'Distinct explicitly supported source facets divided by all source facets; question-given/interpretive excluded.',
            'strict_relevant_chunk_precision_at_5': 'Fraction of Top-5 chunks supporting at least one source facet (not information retrieval precision under official evidence mapping).',
            'mean_reciprocal_rank_at_5': 'Mean reciprocal rank of first chunk that explicitly supports at least one source facet.',
        },
        'per_method': {method: summarise([x for x in details if x['method'] == method]) for method in METHODS},
        'per_question': details,
        'notes': [
            'Provisional AI-assisted annotation; manually adjudicate every label before research publication.',
            'Not official GraphRAG-Bench Evidence Recall; facets were derived from reference evidence separated by semicolons.',
            'Partial facets do not count toward strict coverage.',
            'Question-given facts and interpretive facets were excluded from retrieval scoring.',
            'Scores on 12 deliberately balanced questions are descriptive, not statistically representative.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--rubric', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-human-validated', action='store_true')
    args = parser.parse_args()
    with args.annotations.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    report = qualify(read_json(args.pilot), read_json(args.rubric), rows,
                     require_human=args.require_human_validated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Status:', report['status'])
    for method, figures in report['per_method'].items():
        print(method, figures)
    print('Output:', args.output)


if __name__ == '__main__':
    main()
