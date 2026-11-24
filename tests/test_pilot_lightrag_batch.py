"""CPU-only tests for the batch snapshot and fail-closed resume."""
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_lightrag_batch import check_state, digest, publish, restore_latest


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


if __name__ == "__main__":
    unittest.main()
