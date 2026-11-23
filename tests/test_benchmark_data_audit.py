"""Offline tests for tools/benchmark_data_audit.py."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_data_audit import TYPES, audit, corpus_sources


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("Corpus", "Questions"):
            (self.root / "Datasets" / folder).mkdir(parents=True)
        self.questions = {}
        for domain in ("medical", "novel"):
            source = "Medical" if domain == "medical" else "Novel-1"
            corpus = {"corpus_name": source, "context": "Example document."}
            self.write("Corpus", f"{domain}.json", corpus if domain == "medical" else [corpus])
            self.questions[domain] = [
                {"id": f"{domain}-{kind_index}-{j}",
                 "source": source, "question": f"{domain}: {kind}, question {j}?",
                 "answer": f"Answer {j}", "evidence": "Evidence",
                 "question_type": kind}
                for kind_index, kind in enumerate(TYPES) for j in range(4)
            ]
        self.save()

    def write(self, folder, filename, content):
        (self.root / "Datasets" / folder / filename).write_text(
            json.dumps(content), encoding="utf-8"
        )

    def save(self):
        for domain, rows in self.questions.items():
            self.write("Questions", f"{domain}_questions.json", rows)

    def test_corpus_formats(self):
        self.assertEqual(corpus_sources("medical", {"corpus_name": "Medical", "context": "x"}), {"Medical"})
        self.assertEqual(corpus_sources("novel", [{"corpus_name": "Novel-1", "context": "x"}]), {"Novel-1"})

    def test_reproducible_balanced_pilot(self):
        report, pilot = audit(self.root, seed=42)
        _, again = audit(self.root, seed=42)
        self.assertEqual(report["rows_checked"], 32)
        self.assertEqual(report["excluded_rows"], 0)
        self.assertEqual(pilot, again)
        self.assertEqual(len(pilot), 12)
        self.assertEqual(len({q["id"] for q in pilot}), 12)
        for domain in ("medical", "novel"):
            self.assertEqual(list(report["pilot_distribution"][domain].values()), [2, 2, 1, 1])

    def test_collision_excludes_both(self):
        self.questions["novel"][1]["id"] = self.questions["novel"][0]["id"]
        self.save()
        report, pilot = audit(self.root)
        self.assertEqual(report["issue_counts"]["duplicate_id"], 2)
        self.assertEqual(report["excluded_rows"], 2)
        self.assertEqual(len(report["duplicate_ids"]), 1)
        self.assertNotIn(self.questions["novel"][0]["id"], {q["id"] for q in pilot})

    def test_unknown_source(self):
        self.questions["medical"][0]["source"] = "absent"
        self.save()
        report, _ = audit(self.root)
        self.assertEqual(report["issue_counts"]["source_not_in_corpus"], 1)

    def test_conflicting_answers_for_same_question(self):
        self.questions["novel"][1]["question"] = self.questions["novel"][0]["question"]
        self.questions["novel"][1]["answer"] = "Contradictory reference"
        self.save()
        report, _ = audit(self.root)
        self.assertEqual(report["issue_counts"]["conflicting_reference"], 2)

    def test_malformed_questions(self):
        self.write("Questions", "novel_questions.json", {"unexpected": "object"})
        with self.assertRaises(ValueError):
            audit(self.root)


if __name__ == "__main__":
    unittest.main()
