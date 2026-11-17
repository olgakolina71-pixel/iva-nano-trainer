import unittest
from unittest.mock import patch

from iva_nano.web import _TextAndLinks, _validate_public_url, format_web_context


class WebSafetyTests(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        with self.assertRaises(ValueError):
            _validate_public_url("file:///etc/passwd")

    def test_rejects_localhost(self):
        with self.assertRaises(ValueError):
            _validate_public_url("http://localhost/")

    @patch("iva_nano.web.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("93.184.216.34", 0))
    ])
    def test_accepts_public_http_host(self, _dns):
        self.assertEqual(_validate_public_url("https://example.com/a"), "https://example.com/a")

    def test_html_parser_skips_scripts_and_extracts_links(self):
        parser = _TextAndLinks()
        parser.feed('<title>Demo</title><p>Hello world</p><script>ignore me</script><a href="/next">Next</a>')
        self.assertEqual(parser.title(), "Demo")
        self.assertIn("Hello world", parser.text())
        self.assertNotIn("ignore me", parser.text())
        self.assertEqual(parser.links[0][0], "/next")

    def test_web_context_marks_pages_untrusted(self):
        context = format_web_context([], [{"title": "x", "url": "https://example.com", "text": "data"}])
        self.assertIn("недоверенный текст", context)
        self.assertIn("https://example.com", context)


if __name__ == "__main__":
    unittest.main()
