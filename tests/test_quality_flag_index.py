import json
import tempfile
import unittest
from pathlib import Path

from scripts.index_quality_flags import inspect


class QualityFlagIndexTests(unittest.TestCase):
    def test_indexes_only_line_numbers_and_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "candidate.txt"
            index = root / "reports" / "flags.jsonl"
            report_path = root / "reports" / "summary.json"
            source.write_text(
                "Обычная строка\n"
                + ("А" * 12) + "\n"
                + ("Б" * 10001) + "\n"
                + "ошибка \ufffd кодировки\n",
                encoding="utf-8",
            )

            report = inspect(source, index, report_path)
            index_text = index.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in index_text.splitlines()]

        self.assertEqual(report["records_scanned"], 4)
        self.assertEqual(report["flagged_records"], 3)
        self.assertEqual(rows[0], {"line": 2, "flags": ["long_character_repeat"]})
        self.assertEqual(rows[1]["line"], 3)
        self.assertIn("over_10000_characters", rows[1]["flags"])
        self.assertEqual(rows[2], {"line": 4, "flags": ["replacement_character"]})
        self.assertFalse(report["training_approved"])
        self.assertNotIn("Б" * 10, index_text)


if __name__ == "__main__":
    unittest.main()
