"""Tests of exact chunk preservation, independent of LightRAG and GPU."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_lightrag_real_chunks import single_chunk, check_storage


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class RealChunksTests(unittest.TestCase):
    def test_does_not_resplit_or_normalize(self):
        text = "First  line.\nSecond line."
        self.assertEqual(single_chunk(FakeTokenizer(), text, None, False, 32, 2),
                         [{"tokens": 4, "content": text, "chunk_order_index": 0}])

    def test_empty_chunk_rejected(self):
        with self.assertRaises(ValueError):
            single_chunk(FakeTokenizer(), " ", None, False, 32, 256)

    def test_provenance_and_content(self):
        row = {"content": "a", "full_doc_id": "doc-1", "file_path": "source/chunk-1"}
        check_storage(row, "a", "doc-1", "source/chunk-1")
        for changed in ("content", "full_doc_id", "file_path"):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    check_storage({**row, changed: "wrong"}, "a", "doc-1", "source/chunk-1")

    def test_missing_chunk_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            check_storage(None, "a", "doc-1", "source/chunk-1")


if __name__ == "__main__":
    unittest.main()
