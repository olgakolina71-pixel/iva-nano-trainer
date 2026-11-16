import json
import tempfile
import unittest
from pathlib import Path

from audit_candidate_corpus import audit


class CandidateAuditTests(unittest.TestCase):
    def test_audit_reports_signals_without_approving_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "quarantine"
            root.mkdir()
            data = root / "sample.txt"
            data.write_text(
                "Это русский текст для проверки.\n\n"
                "Это русский текст для проверки.\n\n"
                "hello world test@example.com",
                encoding="utf-8",
            )
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({"datasets": [{
                "id": "sample", "path": str(data),
                "use": "pending_rights_review", "rights_status": "unverified",
            }]}), encoding="utf-8")
            report = audit(manifest, root)
        source = report["sources"][0]
        self.assertEqual(report["status"], "quarantine_only")
        self.assertEqual(source["exact_duplicates"], 1)
        self.assertGreaterEqual(source["possible_email_records"], 1)
        self.assertTrue(source["review_required"])
        self.assertEqual(source["use"], "pending_rights_review")


if __name__ == "__main__":
    unittest.main()
