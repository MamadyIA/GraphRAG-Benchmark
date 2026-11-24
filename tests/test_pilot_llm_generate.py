"""Offline tests of LLM prompt isolation, aligned inputs, checkpoint and resume."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_llm_generate import (
    paired_inputs, messages_for, generate_paired, export_review,
)


class LLMGenerationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.question = {
            "id": "M-1", "domain": "medical", "source": "Medical",
            "question_type": "Fact Retrieval",
            "question": "What is the stage of this disease?",
            "ground_truth": "SECRET_REFERENCE_ANSWER",
            "evidence": "SECRET_REFERENCE_EVIDENCE",
        }
        self.hits = [
            {"rank": i, "chunk_index": i + 10,
             "score": 0.1 * i, "text": f"Documentary evidence passage number {i}."}
            for i in range(1, 6)
        ]
        for mode in ("tfidf_shared", "bge_shared"):
            self.write_input(mode, [self.make_record(mode)])

    def make_record(self, method):
        return {**self.question, "context": [h["text"] for h in self.hits],
                "retrieved_chunks": [dict(h) for h in self.hits],
                "retrieval_mode": method, "generated_answer": None}

    def write_input(self, method, rows):
        (self.root / f"pilot_{method}_retrieval.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

    def test_paired_inputs(self):
        pairs = paired_inputs(self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["retrieval_mode"], "tfidf_shared")

    def test_reference_not_in_prompt(self):
        row = self.make_record("tfidf_shared")
        messages = messages_for(row)
        prompt = json.dumps(messages)
        self.assertNotIn("SECRET_REFERENCE_ANSWER", prompt)
        self.assertNotIn("SECRET_REFERENCE_EVIDENCE", prompt)
        self.assertIn("[P1]", prompt)
        self.assertIn("What is the stage", prompt)

    def test_mismatched_question_fails(self):
        other = self.make_record("bge_shared")
        other["question"] = "Wrong question"
        self.write_input("bge_shared", [other])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            paired_inputs(self.root)

    def test_wrong_shared_chunk_text_fails(self):
        other = self.make_record("bge_shared")
        other["retrieved_chunks"][0]["text"] = "Contradiction"
        other["context"][0] = "Contradiction"
        self.write_input("bge_shared", [other])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            paired_inputs(self.root)

    def test_checkpoint_resume_and_review(self):
        pairs = paired_inputs(self.root)
        path = self.root / "checkpoint.json"
        calls = []
        def fake_infer(messages, max_new_tokens, max_input_tokens):
            calls.append(messages)
            return {"answer": "Stage zero [P1].", "generated_tokens": 6, "input_tokens": 60}
        state = generate_paired(pairs, fake_infer, path, "fake")
        self.assertEqual(len(calls), 2)
        self.assertEqual(state["status"], "COMPLETE")
        state = generate_paired(pairs, fake_infer, path, "fake")
        self.assertEqual(len(calls), 2, "completed work must not repeat")
        review = self.root / "review.csv"
        export_review(pairs, state, review)
        with review.open(encoding="utf-8-sig", newline="") as f:
            data = list(csv.DictReader(f))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["reference_answer"], "SECRET_REFERENCE_ANSWER")
        self.assertEqual(data[0]["tfidf_answer"], "Stage zero [P1].")
        self.assertFalse(data[0]["bge_correctness_yes_partial_no"])

    def test_checkpoint_rejects_different_model(self):
        pairs = paired_inputs(self.root)
        path = self.root / "state.json"
        def infer(messages, max_new_tokens, max_input_tokens):
            return {"answer": "Generated."}
        generate_paired(pairs, infer, path, "model-a")
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
            generate_paired(pairs, infer, path, "model-b")

    def test_missing_top_five_fails(self):
        bad = self.make_record("bge_shared")
        bad["retrieved_chunks"] = bad["retrieved_chunks"][:-1]
        bad["context"] = bad["context"][:-1]
        self.write_input("bge_shared", [bad])
        with self.assertRaisesRegex(ValueError, "Top-5"):
            paired_inputs(self.root)

    def test_empty_llm_response_fails_closed(self):
        pairs = paired_inputs(self.root)
        with self.assertRaisesRegex(ValueError, "empty or invalid"):
            generate_paired(pairs, lambda *_: {"answer": ""}, self.root / "state.json", "fake")


if __name__ == "__main__":
    unittest.main()
