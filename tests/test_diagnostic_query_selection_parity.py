import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.cue_cases import select_queries_for_cases
from diagnostic.data_loading import QueryRecord


class DiagnosticQuerySelectionParityTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            QueryRecord(0, "person wearing a red shirt", 1),
            QueryRecord(1, "person wearing a blue shirt", 2),
            QueryRecord(2, "person wearing a red shirt and blue shirt", 3),
        ]
        self.gallery_pids = np.asarray([1, 2, 3], dtype=np.int64)

    def test_plain_case_requires_all_case_needles(self):
        cases = [{"case_id": "red_blue", "cue_a": "red shirt", "cue_b": "blue shirt"}]

        selected, skipped = select_queries_for_cases(
            "toy",
            cases,
            self.records,
            self.gallery_pids,
            max_queries_per_case=None,
        )

        self.assertEqual([row["query_id"] for row in selected], [2])
        self.assertEqual(skipped, [])

    def test_query_regex_controls_regex_only_case_selection(self):
        cases = [
            {
                "case_id": "red_or_blue",
                "cue_a": "red shirt",
                "cue_b": "blue shirt",
                "query_regex": r"\b(?:red|blue)\s+shirt\b",
            }
        ]

        selected, skipped = select_queries_for_cases(
            "toy",
            cases,
            self.records,
            self.gallery_pids,
            max_queries_per_case=None,
        )

        self.assertEqual([row["query_id"] for row in selected], [0, 1, 2])
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()

