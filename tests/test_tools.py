import unittest
from unittest.mock import patch

from iva_nano.tools import calculate, run_tool, format_tool_result


class CalculatorTests(unittest.TestCase):
    def test_basic_operations_and_parentheses(self):
        self.assertEqual(calculate("2 + 3 * 4"), 14)
        self.assertEqual(calculate("(2 + 3) * 4"), 20)
        self.assertEqual(calculate("20 / 5"), 4)

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

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            run_tool("shell", {"command": "echo unsafe"})


if __name__ == "__main__":
    unittest.main()
