import unittest
from unittest.mock import patch

from iva_nano.tools import calculate, run_tool, format_tool_result, select_tool


class CalculatorTests(unittest.TestCase):
    def test_basic_operations_and_parentheses(self):
        self.assertEqual(calculate("2 + 3 * 4"), 14)
        self.assertEqual(calculate("(2 + 3) * 4"), 20)
        self.assertEqual(calculate("20 / 5"), 4)
        self.assertEqual(calculate("20 : 5"), 4)

    def test_rejects_code_and_unknown_syntax(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('echo unsafe')")
        with self.assertRaises(ValueError):
            calculate("2 ** 100")

    def test_division_by_zero_is_friendly_error(self):
        with self.assertRaisesRegex(ValueError, "ноль"):
            calculate("5 / 0")

    def test_time_tool_returns_timezone_aware_value(self):
        result = run_tool("current_time")
        self.assertIn("+", result["result"])

    @patch("iva_nano.tools.search_wikipedia")
    def test_search_tool_returns_sources(self, mocked_search):
        mocked_search.return_value = [{
            "title": "Тест", "extract": "Короткая справка.",
            "url": "https://ru.wikipedia.org/wiki/Тест",
        }]
        result = run_tool("web_search", {"query": "тест"})
        self.assertEqual(result["results"][0]["title"], "Тест")
        self.assertIn("Источник:", format_tool_result(result))

    def test_automatic_selection_routes_math_time_and_search(self):
        self.assertEqual(
            select_tool("Сколько будет 7 * 8?")["name"], "calculator"
        )
        self.assertEqual(
            select_tool("Который час?")["name"], "current_time"
        )
        selected = select_tool("Найди в интернете историю космонавтики")
        self.assertEqual(selected["name"], "web_search")
        self.assertIn("историю космонавтики", selected["arguments"]["query"])

    def test_automatic_selection_leaves_small_talk_to_the_model(self):
        self.assertIsNone(select_tool("Привет, как дела?"))
        self.assertIsNone(select_tool("Расскажи шутку"))

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            run_tool("shell", {"command": "echo unsafe"})


if __name__ == "__main__":
    unittest.main()
