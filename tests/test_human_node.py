import unittest
from unittest.mock import patch

from nodes.human_node import _parse_user_input


class HumanNodeParsingTests(unittest.TestCase):
    def setUp(self):
        self.findings = [
            {"id": 1, "subtopic": "AI ethics", "summary": "", "key_points": [], "sources": []},
            {"id": 2, "subtopic": "AI safety", "summary": "", "key_points": [], "sources": []},
            {"id": 3, "subtopic": "AI applications", "summary": "", "key_points": [], "sources": []},
        ]

    def test_approve_and_modify_returns_sorted_items(self):
        approved = _parse_user_input(self.findings, 'approve 3,1 | modify 3 "Applied AI"')
        self.assertEqual([item["id"] for item in approved], [1, 3])
        self.assertEqual(approved[1]["subtopic"], "Applied AI")

    def test_reject_then_fallback_approves_remaining_pool(self):
        approved = _parse_user_input(self.findings, "reject 2")
        self.assertEqual([item["id"] for item in approved], [1, 3])

    def test_invalid_approve_tokens_fallbacks_to_all(self):
        approved = _parse_user_input(self.findings, "approve a,b")
        self.assertEqual([item["id"] for item in approved], [1, 2, 3])

    def test_add_duplicate_is_ignored(self):
        with patch("nodes.human_node.quick_search", return_value=[]), patch(
            "nodes.human_node.synthesize_subtopic_from_results",
            return_value={"summary": "", "key_points": [], "sources": []},
        ):
            approved = _parse_user_input(self.findings, 'add "AI ethics" | approve 1')
        self.assertEqual([item["id"] for item in approved], [1])


if __name__ == "__main__":
    unittest.main()
