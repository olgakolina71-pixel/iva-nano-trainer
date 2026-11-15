import json
import tempfile
import unittest
from pathlib import Path

from prepare_candidate_corpus import has_personal_markers, prepare, source_documents


class CandidateCorpusTests(unittest.TestCase):
    def test_json_arrays_are_decoded_and_rights_stay_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "education.txt").write_text(
                '["Заголовок", "Это достаточно длинный русский текст для проверки нормализации."]\n'
                '["Заголовок", "Это достаточно длинный русский текст для проверки нормализации."]\n',
                encoding="utf-8",
            )
            manifest = root / "input.json"
            manifest.write_text(json.dumps({"datasets": [{
                "id": "ru_edutexts_cc0",
                "path": "data/raw/hf/education.txt",
                "license": "CC0-1.0",
                "attribution": "source attribution",
                "use": "pending_rights_review",
                "rights_status": "dataset_card_claim_unverified",
            }]}), encoding="utf-8")
            out = root / "normalized"
            out_manifest = out / "manifest.json"
            prepare(raw, manifest, out, out_manifest)
            result = json.loads(out_manifest.read_text(encoding="utf-8"))["datasets"][0]
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["preparation"]["duplicates_removed"], 1)
            self.assertEqual(result["use"], "pending_rights_review")
            self.assertEqual(result["rights_status"], "dataset_card_claim_unverified")
            text = (out / "ru_edutexts_cc0.txt").read_text(encoding="utf-8")
            self.assertIn("Это достаточно длинный русский текст", text)
            self.assertNotIn("[", text)

    def test_dialogue_turns_stay_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dialogue.txt"
            path.write_text(
                "### Пользователь:\nПривет\n### IVA:\nЗдравствуйте\n\n",
                encoding="utf-8",
            )
            docs = list(source_documents(path, "small_life_dialogues_cc0"))
            self.assertEqual(len(docs), 1)
            self.assertIn("### Пользователь:", docs[0])
            self.assertIn("### IVA:", docs[0])

    def test_obvious_student_identifiers_are_flagged(self):
        self.assertTrue(has_personal_markers("Презентацию подготовила Строковой Натальи 10 «Б»."))
        self.assertTrue(has_personal_markers("Контакт: test@example.com"))
        self.assertFalse(has_personal_markers("Объяснение принципа работы нейронной сети."))


if __name__ == "__main__":
    unittest.main()
