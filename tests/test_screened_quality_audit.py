import tempfile
import unittest
from pathlib import Path

from scripts.audit_screened_candidate import audit


class ScreenedCandidateAuditTests(unittest.TestCase):
    def test_reports_aggregate_quality_without_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "candidate.txt"
            source.write_text(
                "Привет, это русский текст.\n"
                "English text remains.\n"
                "bad email sample@example.com\n"
                + ("А" * 12)
                + "\n",
                encoding="utf-8",
            )
            report = audit(source)

        metrics = report["metrics"]
        self.assertEqual(report["status"], "candidate_quality_audit_only")
        self.assertFalse(report["training_approved"])
        self.assertEqual(metrics["records"], 4)
        self.assertEqual(metrics["possible_email_records_remaining"], 1)
        self.assertEqual(metrics["long_character_repeat_records"], 1)
        self.assertGreater(metrics["cyrillic_share_of_letters"], 0)


if __name__ == "__main__":
    unittest.main()
