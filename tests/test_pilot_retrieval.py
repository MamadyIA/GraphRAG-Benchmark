"""Offline tests for retrieval-only pilot prototype (no model download)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_retrieval import LexicalIndex, corpus_map, load_pilot, retrieve, split_words


class PilotRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("Datasets/Corpus", "Datasets/Questions"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "Datasets/Corpus/medical.json").write_text(
            json.dumps({"corpus_name": "Medical", "context":
                        "Basal cell carcinoma is a skin cancer. " * 30}),
            encoding="utf-8"
        )
        (self.root / "Datasets/Corpus/novel.json").write_text(
            json.dumps([{"corpus_name": "Novel-1", "context":
                         "Suza lives in Port William. Aunt Katy raised Suza. " * 30}]),
            encoding="utf-8"
        )
        self.pilot = [
            {"id": "M-1", "domain": "medical", "source": "Medical",
             "question_type": "Fact Retrieval", "question": "What is basal cell carcinoma?",
             "answer": "A cancer.", "evidence": "Basal cell carcinoma is a skin cancer."},
            {"id": "N-1", "domain": "novel", "source": "Novel-1",
             "question_type": "Complex Reasoning", "question": "Who raised Suza?",
             "answer": "Aunt Katy.", "evidence": "Aunt Katy raised Suza."}
        ]
        self.pilot_path = self.root / "pilot.json"
        self.save()

    def save(self):
        self.pilot_path.write_text(json.dumps(self.pilot), encoding="utf-8")

    def test_chunks_and_overlap(self):
        self.assertEqual(split_words("a b c d e f", 4, 2),
                         ["a b c d", "c d e f", "e f"])
        self.assertEqual(split_words("", 4, 2), [])

    def test_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            split_words("abc", 3, 3)
        with self.assertRaises(ValueError):
            split_words("abc", 0, 0)

    def test_corpus_keys(self):
        corpora = corpus_map(self.root)
        self.assertEqual(set(corpora), {("medical", "Medical"), ("novel", "Novel-1")})

    def test_pilot_validation(self):
        self.assertEqual(len(load_pilot(self.pilot_path)), 2)
        self.pilot[1]["id"] = self.pilot[0]["id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate pilot id"):
            load_pilot(self.pilot_path)

    def test_lexical_ranks_relevant_chunk_first(self):
        index = LexicalIndex(["orange banana fruit", "aunt katy raised suza"])
        matches = index.search("who raised suza aunt katy", 2)
        self.assertEqual(matches[0]["chunk_index"], 1)
        self.assertGreater(matches[0]["score"], matches[1]["score"])

    def test_retrieval_preserves_question_ids(self):
        results = retrieve(self.root, self.pilot, mode="lexical",
                           chunk_size=12, overlap=2, top_k=3)
        self.assertEqual([r["id"] for r in results], ["M-1", "N-1"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(0 < len(r["context"]) <= 3 for r in results))
        self.assertTrue(all(r["generated_answer"] is None for r in results))
        self.assertEqual(results[0]["ground_truth"], "A cancer.")

    def test_missing_source_fails_closed(self):
        self.pilot[1]["source"] = "Nonexistent"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            retrieve(self.root, self.pilot)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            retrieve(self.root, self.pilot, mode="unsupported")


if __name__ == "__main__":
    unittest.main()
