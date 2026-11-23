"""Deterministic smoke tests of shared-chunk retrieval and review files."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_fair_compare import compare, write_outputs


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.lower().split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeModel:
    tokenizer = FakeTokenizer()

    def encode(self, texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for item in texts:
            item = item.lower()
            vec = [float(item.count("surgery")), float(item.count("cancer")),
                   float(item.count("suza")), 1.0]
            norm = np.linalg.norm(vec)
            vectors.append(np.array(vec, dtype=np.float32) / norm)
        return np.stack(vectors)


class FairCompareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "Datasets/Corpus").mkdir(parents=True)
        (self.root / "Datasets/Corpus/medical.json").write_text(json.dumps({
            "corpus_name": "Medical",
            "context": ("Surgery helps treat cancer and removes tumors. " * 12)
        }), encoding="utf-8")
        (self.root / "Datasets/Corpus/novel.json").write_text(json.dumps([{
            "corpus_name": "Novel-1",
            "context": ("Suza lives by the river with Aunt Katy. " * 12)
        }]), encoding="utf-8")
        self.pilot = [
            {"id": "M-1", "domain": "medical", "source": "Medical",
             "question_type": "Fact Retrieval", "question": "Does surgery treat cancer?",
             "answer": "Yes", "evidence": "Surgery helps treat cancer."},
            {"id": "N-1", "domain": "novel", "source": "Novel-1",
             "question_type": "Fact Retrieval", "question": "Where does Suza live?",
             "answer": "By the river", "evidence": "Suza lives by the river."},
        ]

    def test_shared_candidate_indices_and_lengths(self):
        output, summary, sources = compare(self.root, self.pilot, FakeModel(),
                                           chunk_size=12, overlap=3, top_k=2)
        self.assertEqual(sources, 2)
        self.assertEqual(len(summary), 2)
        for method in ("tfidf_shared", "bge_shared"):
            self.assertEqual(len(output[method]), 2)
            for item in output[method]:
                self.assertEqual(len(item["retrieved_chunks"]), 2)
                self.assertTrue(all(hit["chunk_index"] >= 0
                                    for hit in item["retrieved_chunks"]))
                self.assertIsNone(item["generated_answer"])
        self.assertEqual(
            output["tfidf_shared"][0]["id"],
            output["bge_shared"][0]["id"],
        )

    def test_review_file_contains_unlabeled_chunks(self):
        output, summary, sources = compare(self.root, self.pilot, FakeModel(),
                                           chunk_size=12, overlap=3, top_k=2)
        files, report, review = write_outputs(
            self.root / "out", output, summary, sources, "fake",
            12, 3, 2
        )
        self.assertTrue(report.is_file())
        self.assertEqual(set(files), {"tfidf_shared", "bge_shared"})
        with review.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(not row["supports_evidence_yes_no_partial"]
                            for row in rows))
        self.assertTrue(all(row["retrieved_passage"] for row in rows))

    def test_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            compare(self.root, self.pilot, FakeModel(), top_k=0)

    def test_unknown_pilot_source_fails(self):
        self.pilot[0]["source"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown source"):
            compare(self.root, self.pilot, FakeModel(),
                    chunk_size=12, overlap=3, top_k=2)


if __name__ == "__main__":
    unittest.main()
