import unittest

from iva_nano.emotion import EmotionEngine


class EmotionEngineTests(unittest.TestCase):
    def test_positive_and_curious_cues_change_mood(self):
        engine = EmotionEngine()
        self.assertEqual(engine.observe("Спасибо, получилось круто!")["mood"], "радость")
        self.assertEqual(engine.observe("Интересно, почему так?")["mood"], "интерес")

    def test_repeated_hostility_sets_a_resumable_boundary(self):
        engine = EmotionEngine(pause_threshold=2)
        engine.observe("Ты идиот")
        state = engine.observe("Заткнись")
        self.assertTrue(state["paused"])
        engine.resume()
        self.assertFalse(engine.status()["paused"])
        self.assertEqual(engine.status()["hostile_turns"], 0)

    def test_context_is_clear_that_emotions_are_simulated(self):
        self.assertIn("НЕ НАСТОЯЩИЕ ПЕРЕЖИВАНИЯ", EmotionEngine().context_for())

    def test_mood_automatically_changes_response_style_hint(self):
        engine = EmotionEngine()
        engine.observe("Спасибо, получилось круто!")
        context = engine.context_for()
        self.assertIn("чуть теплее и энергичнее", context)
        self.assertIn("не вставляй его название в каждый ответ", context)

    def test_reset_returns_to_neutral(self):
        engine = EmotionEngine()
        engine.observe("Спасибо!")
        engine.reset()
        self.assertEqual(engine.status()["mood"], "спокойствие")
        self.assertFalse(engine.status()["paused"])


if __name__ == "__main__":
    unittest.main()
