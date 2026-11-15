import tempfile
import unittest
from pathlib import Path

from prepare_auxiliary_text import association_lines, sentiment_lines, prepare


class PreparationTests(unittest.TestCase):
    def test_association_rows_become_russian_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assoc.csv"
            path.write_text("word;assoc;is_safe\nкот;животное;1\n", encoding="utf-8")
            lines = list(association_lines(path))
            self.assertTrue(any("кот" in line and "животное" in line for line in lines))

    def test_sentiment_rows_become_russian_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sent.csv"
            path.write_text("term;tag;value\nрадость;PSTV;0.8\n", encoding="utf-8")
            lines = list(sentiment_lines(path))
            self.assertTrue(any("радость" in line and "положительную" in line for line in lines))

    def test_preparation_writes_manifest_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assoc = root / "assoc.csv"
            assoc.write_text("word;assoc;is_safe\nкот;животное;1\n", encoding="utf-8")
            result = prepare(str(assoc), None, root / "raw", root / "manifest.json")
            self.assertTrue(Path(result["text"]).exists())
            self.assertTrue(Path(result["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
