import json
import tempfile
import unittest
from pathlib import Path

from iva_nano.subconscious import Subconscious


class SubconsciousTests(unittest.TestCase):
    def test_reflection_detects_repeated_topic_without_persisting_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            mind = Subconscious(memory_path)
            mind.reflect("Давай обсудим проект", "Хорошо")
            notes = mind.reflect("Вернёмся к проекту?", "Давай")
            self.assertTrue(any("проект" in note for note in notes))
            self.assertFalse(memory_path.exists())

    def test_memory_is_saved_only_after_explicit_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            mind = Subconscious(path)
            mind.reflect("Мне нравится физика", "Понял")
            self.assertFalse(path.exists())
            mind.remember("Пользователь изучает физику")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["memories"], ["Пользователь изучает физику"])
            reloaded = Subconscious(path)
            self.assertEqual(reloaded.memory_list(), ["Пользователь изучает физику"])

    def test_forget_all_removes_saved_memory_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            mind = Subconscious(path)
            mind.remember("Заметка")
            mind.forget_all()
            self.assertFalse(path.exists())
            self.assertEqual(mind.memory_list(), [])

    def test_memory_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            mind = Subconscious(Path(tmp) / "memory.json", max_memories=2)
            mind.remember("one")
            mind.remember("two")
            mind.remember("three")
            self.assertEqual(mind.memory_list(), ["two", "three"])


if __name__ == "__main__":
    unittest.main()
