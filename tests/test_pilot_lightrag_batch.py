"""CPU-only tests for the batch snapshot and fail-closed resume."""
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_lightrag_batch import (check_state, digest, publish, restore_latest,
                                  bounded_generation, classify_generation,
                                  COMPLETION_MARKER, GENERATION_CAPS, repeated_records,
                                  valid_extraction, EXTRACT_SYSTEM, EXTRACT_USER)


class BatchCheckpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = self.root / "snapshots"
        self.store.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()
        (self.work / "graph.graphml").write_text("fixture", encoding="utf-8")
        self.cfg = {"source": "Novel-test", "total_chunks": 3}
        self.rows = [{"chunk_index": 0}, {"chunk_index": 1}]

    def test_restore_empty(self):
        self.assertEqual(restore_latest(self.store, self.work, self.cfg), [])

    def test_round_trip_and_next_snapshot(self):
        first = publish(self.store, self.work, self.cfg, self.rows)
        self.assertTrue(first.exists())
        restored = self.root / "restored"
        restored.mkdir()
        self.assertEqual(restore_latest(self.store, restored, self.cfg), self.rows)
        self.assertEqual((restored / "graph.graphml").read_text(), "fixture")
        second = publish(self.store, restored, self.cfg, self.rows + [{"chunk_index": 2}])
        self.assertTrue(first.exists() and second.exists())
        another = self.root / "another"
        another.mkdir()
        self.assertEqual(len(restore_latest(self.store, another, self.cfg)), 3)

    def test_wrong_config_fails(self):
        publish(self.store, self.work, self.cfg, self.rows)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            restore_latest(self.store, self.work, {**self.cfg, "source": "different"})

    def test_missing_index_fails(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            check_state({"config": self.cfg, "progress": [{"chunk_index": 1}]},
                        self.cfg, 1)

    def test_corrupt_latest_is_not_silently_skipped(self):
        publish(self.store, self.work, self.cfg, self.rows)
        (self.store / "index_000003.zip").write_bytes(b"invalid")
        with self.assertRaises(Exception):
            restore_latest(self.store, self.work, self.cfg)

    def test_refuses_overwrite(self):
        first = publish(self.store, self.work, self.cfg, self.rows)
        content = first.read_bytes()
        with self.assertRaises(FileExistsError):
            publish(self.store, self.work, self.cfg, self.rows)
        self.assertEqual(first.read_bytes(), content)

    def test_partial_archive_not_restored(self):
        publish(self.store, self.work, self.cfg, self.rows)
        (self.store / ".index_fake.partial").write_bytes(b"unfinished")
        self.assertEqual(restore_latest(self.store, self.work, self.cfg), self.rows)

    def test_unsafe_member_fails(self):
        with ZipFile(self.store / "index_000001.zip", "w") as z:
            z.writestr("batch_state.json", "{}")
            z.writestr("../escape", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            restore_latest(self.store, self.work, self.cfg)

    def test_digest_is_stable(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"a": 1}), digest({"a": 2}))


ENTITY = "entity<|#|>Log School House<|#|>Artifact<|#|>Educational setting."
VALID = ENTITY + "\n" + COMPLETION_MARKER


class GenerationCompletionTests(unittest.TestCase):
    def test_marker_completes_even_at_cap(self):
        self.assertEqual(classify_generation(VALID,
                                             2048, 2048, True, False),
                         "complete_marker")

    def test_retry_length_then_marker(self):
        attempted = []
        def generate(cap, marker):
            attempted.append(cap)
            self.assertTrue(marker)
            return ("entity", cap, False) if len(attempted) == 1 else (
                VALID, 2200, False)
        self.assertTrue(bounded_generation(generate, True).endswith(COMPLETION_MARKER))
        self.assertEqual(attempted, list(GENERATION_CAPS))

    def test_hard_cap_fails_closed(self):
        attempted = []
        def generate(cap, marker):
            attempted.append(cap)
            return "unfinished", cap, False
        with self.assertRaisesRegex(RuntimeError, "token_cap"):
            bounded_generation(generate, True)
        self.assertEqual(attempted, list(GENERATION_CAPS))

    def test_eos_without_extraction_marker_fails(self):
        attempted = []
        def generate(cap, marker):
            attempted.append(cap)
            return "incomplete", 100, True
        with self.assertRaisesRegex(RuntimeError, "missing_completion_marker"):
            bounded_generation(generate, True)
        self.assertEqual(attempted, list(GENERATION_CAPS))

    def test_summary_eos_and_empty_rejected(self):
        self.assertEqual(bounded_generation(lambda cap, mark: ("summary", 10, True), False),
                         "summary")
        with self.assertRaisesRegex(RuntimeError, "empty"):
            bounded_generation(lambda cap, mark: ("", 1, True), True)

    def test_marker_not_at_end_is_not_completion(self):
        self.assertEqual(classify_generation("entity" + COMPLETION_MARKER + "extra",
                                             100, 2048, True, True),
                         "missing_completion_marker")


class AntiLoopTests(unittest.TestCase):
    def test_observed_duplicate_loop_is_rejected(self):
        loop = "\n".join([ENTITY] * 4)
        self.assertTrue(repeated_records(loop))
        self.assertEqual(classify_generation(loop, 500, 2048, True, False), "repetition")

    def test_retry_recovers_without_accepting_partial_records(self):
        attempted = []
        def generate(cap, marker):
            attempted.append(cap)
            return ("\n".join([ENTITY] * 4), 500, False) if len(attempted) == 1 else (
                VALID, 60, False)
        self.assertEqual(bounded_generation(generate, True), VALID)
        self.assertEqual(attempted, list(GENERATION_CAPS))

    def test_duplicate_with_marker_is_not_valid(self):
        self.assertFalse(valid_extraction(ENTITY + "\n" + ENTITY + "\n" + COMPLETION_MARKER))
        self.assertEqual(classify_generation(ENTITY + "\n" + ENTITY + "\n" +
                         COMPLETION_MARKER, 100, 2048, True, True), "invalid_records")

    def test_real_terminator_and_fields_required(self):
        self.assertTrue(valid_extraction(VALID))
        self.assertFalse(valid_extraction(ENTITY))
        self.assertFalse(valid_extraction("entity<|#|>broken\n" + COMPLETION_MARKER))

    def test_valid_relation_and_distinct_entities(self):
        another = "entity<|#|>Alice<|#|>Person<|#|>Alice studied there."
        rel = "relation<|#|>Alice<|#|>Log School House<|#|>education<|#|>Alice studied there."
        self.assertTrue(valid_extraction("\n".join([ENTITY, another, rel, COMPLETION_MARKER])))
        reverse = "relation<|#|>Log School House<|#|>Alice<|#|>education<|#|>Alice studied there."
        self.assertFalse(valid_extraction("\n".join([ENTITY, another, rel, reverse, COMPLETION_MARKER])))

    def test_prompt_format_matches_upstream_placeholders(self):
        kwargs = dict(entity_types="Person, Artifact", tuple_delimiter="<|#|>",
                      completion_delimiter=COMPLETION_MARKER, language="English",
                      examples="unused", input_text="Alice studied at a log school.")
        self.assertIn(COMPLETION_MARKER, EXTRACT_SYSTEM.format(**kwargs))
        self.assertIn(kwargs["input_text"], EXTRACT_USER.format(**kwargs))


if __name__ == "__main__":
    unittest.main()
