import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect_wikipedia import collect


class WikipediaCorpusCollectionTests(unittest.TestCase):
    def _page(self):
        return {
            "title": "Нейронная сеть",
            "extract": "Описание нейронных сетей. " * 20,
            "url": "https://ru.wikipedia.org/wiki/Нейронная_сеть",
            "page_id": 123,
            "revision_id": 456,
            "revision_timestamp": "2026-11-16T10:00:00Z",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }

    @patch("collect_wikipedia.collect_wikipedia_pages")
    def test_collection_saves_attributed_quarantined_page(self, mocked_fetch):
        mocked_fetch.return_value = [self._page()]
        with tempfile.TemporaryDirectory() as tmp:
            added = collect(["нейронная сеть"], tmp, limit=5)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(added), 1)
            entry = manifest["datasets"][0]
            self.assertEqual(entry["use"], "pending_rights_review")
            self.assertEqual(entry["rights_status"], "pending_human_review")
            self.assertEqual(entry["license"], "CC BY-SA 4.0")
            self.assertIn("revision 456", entry["attribution"])
            self.assertIn("Описание", (Path(tmp) / Path(entry["path"]).name).read_text(encoding="utf-8"))

    @patch("collect_wikipedia.collect_wikipedia_pages")
    def test_repeat_collection_does_not_duplicate_same_revision(self, mocked_fetch):
        mocked_fetch.return_value = [self._page()]
        with tempfile.TemporaryDirectory() as tmp:
            collect(["нейронная сеть"], tmp)
            added_again = collect(["нейронная сеть"], tmp)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(added_again, [])
            self.assertEqual(len(manifest["datasets"]), 1)


if __name__ == "__main__":
    unittest.main()
