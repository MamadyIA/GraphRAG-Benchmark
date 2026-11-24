"""Offline tests for explicit evidence annotations and strict facet metrics."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from evidence_eval import qualify


class EvidenceEvalTests(unittest.TestCase):
    def setUp(self):
        self.pilot = [{'id': 'M1', 'domain': 'medical', 'source': 'Medical',
                       'question_type': 'Complex Reasoning',
                       'evidence': 'Proof A.; Proof B.'}]
        self.rubric = [{'id': 'M1', 'facets': [
            {'index': 1, 'text': 'Proof A.', 'scope': 'source'},
            {'index': 2, 'text': 'Proof B.', 'scope': 'source'}]}]
        self.rows = []
        for method in ('tfidf_shared', 'bge_shared'):
            for rank in range(1, 6):
                yes = '1' if rank == 2 and method == 'tfidf_shared' else ''
                partial = '2' if rank == 3 else ''
                self.rows.append({
                    'question_id': 'M1', 'method': method, 'rank': str(rank),
                    'retrieved_passage': 'nonempty',
                    'facet_ids_yes': yes,
                    'facet_ids_partial': partial,
                    'candidate_label': 'yes' if yes else 'partial' if partial else 'no',
                    'review_status': 'ai_prelabel',
                })

    def test_valid_metrics_and_distinct_facet_union(self):
        report = qualify(self.pilot, self.rubric, self.rows)
        t = report['per_method']['tfidf_shared']
        b = report['per_method']['bge_shared']
        self.assertEqual(t['strict_facet_recall_at_5'], .5)
        self.assertEqual(t['strict_any_evidence_hit_at_5'], 1.0)
        self.assertEqual(t['mean_reciprocal_rank_at_5'], .5)
        self.assertEqual(b['strict_facet_recall_at_5'], 0.0)
        self.assertEqual(b['strict_any_evidence_hit_at_5'], 0.0)
        self.assertIn('PROVISIONAL', report['status'])

    def test_human_validation_guard(self):
        with self.assertRaisesRegex(ValueError, 'human-validated'):
            qualify(self.pilot, self.rubric, self.rows, require_human=True)

    def test_reject_duplicate_annotation_row(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            qualify(self.pilot, self.rubric, self.rows + [self.rows[0]])

    def test_reject_unscorable_facet(self):
        self.rubric[0]['facets'][0]['scope'] = 'question_given'
        with self.assertRaisesRegex(ValueError, 'non-source facets'):
            qualify(self.pilot, self.rubric, self.rows)

    def test_detect_missing_rows(self):
        with self.assertRaisesRegex(ValueError, 'expected 10'):
            qualify(self.pilot, self.rubric, self.rows[:-1])

    def test_reject_mismatched_official_evidence(self):
        self.rubric[0]['facets'][0]['text'] = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'differ from'):
            qualify(self.pilot, self.rubric, self.rows)

    def test_deny_labels_without_facet_support(self):
        self.rows[0]['candidate_label'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'label/facet mismatch'):
            qualify(self.pilot, self.rubric, self.rows)


if __name__ == '__main__':
    unittest.main()
