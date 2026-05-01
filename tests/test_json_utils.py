import unittest

from json_utils import parse_json_robust


class JsonUtilsTests(unittest.TestCase):
    def test_parses_markdown_wrapped_json(self):
        text = """```json
{
  "a": 1,
  "b": [1, 2]
}
```"""
        parsed = parse_json_robust(text, fallback={})
        self.assertEqual(parsed["a"], 1)
        self.assertEqual(parsed["b"], [1, 2])

    def test_repairs_trailing_commas(self):
        text = '{"topic":"ai","subtopics":[{"id":1,}],}'
        parsed = parse_json_robust(text, fallback={"topic": "", "subtopics": []})
        self.assertEqual(parsed["topic"], "ai")
        self.assertEqual(parsed["subtopics"][0]["id"], 1)

    def test_uses_fallback_for_empty(self):
        fallback = {"ok": False}
        parsed = parse_json_robust("", fallback=fallback)
        self.assertEqual(parsed, fallback)


if __name__ == "__main__":
    unittest.main()
