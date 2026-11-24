import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pilot_lightrag_smoke import validate_query


class SmokeValidationTests(unittest.TestCase):
    def test_requires_all_three_retrieval_types(self):
        result = {"status": "success", "data": {
            "entities": [{"name": "A"}],
            "relationships": [{"source": "A", "target": "B"}],
            "chunks": [{"content": "A knows B."}],
        }}
        self.assertEqual(validate_query(result),
                         {"entities": 1, "relationships": 1, "chunks": 1})

    def test_rejects_vector_only_result(self):
        with self.assertRaisesRegex(RuntimeError, "requires retrieved"):
            validate_query({"status": "success", "data": {
                "entities": [], "relationships": [],
                "chunks": [{"content": "A knows B."}],
            }})

    def test_rejects_failure(self):
        with self.assertRaisesRegex(RuntimeError, "did not succeed"):
            validate_query({"status": "failure"})

    def test_rejects_bad_schema(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            validate_query({"status": "success", "data": {
                "entities": {}, "relationships": [], "chunks": [],
            }})


if __name__ == "__main__":
    unittest.main()
