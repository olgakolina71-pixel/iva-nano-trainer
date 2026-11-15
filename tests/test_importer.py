import csv
import io
import unittest

from iva_nano.importer import ALLOWED_DATASETS, EXPECTED_LICENSE, _validate_csv


class ImporterTests(unittest.TestCase):
    def test_allowlist_contains_only_reviewed_sources(self):
        self.assertEqual(len(ALLOWED_DATASETS), 4)

    def test_expected_license_is_noncommercial_sharealike(self):
        self.assertEqual(EXPECTED_LICENSE, "CC BY-NC-SA 4.0")

    def test_valid_csv_is_counted(self):
        raw = "word,label\nкот,positive\nдом,neutral\n".encode("utf-8")
        self.assertEqual(_validate_csv(raw, "test"), 3)

    def test_empty_csv_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_csv(b"", "test")


if __name__ == "__main__":
    unittest.main()
