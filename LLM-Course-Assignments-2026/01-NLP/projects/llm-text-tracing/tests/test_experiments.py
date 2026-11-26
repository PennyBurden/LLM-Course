import unittest

from qa.context_sweep import CASES, POSITIONS, material
from review_experiment import random_review, summarize


class ExperimentTest(unittest.TestCase):
    def test_review_counts_include_equal_scores_and_keep_denominators_separate(self):
        rows = [[42, 0, 0, .90], [8, 0, 2, .70], [3, 1, 1, .60], [12, 2, 1, .45]]
        counts = summarize(rows, .7)
        self.assertEqual((counts["n"], counts["auto"], counts["auto_correct"]), (4, 2, 1))
        self.assertEqual((counts["errors"], counts["review_errors"]), (2, 1))
        self.assertEqual(counts["by_class"][1]["auto"], 0)
        self.assertEqual(random_review(counts), (1, 0, 2))
        self.assertEqual(random_review(summarize(rows, 0)), (0, 0, 0))
        self.assertEqual(random_review(summarize(rows, 1)), (2, 2, 2))

    def test_position_conditions_preserve_all_filler_and_distractors(self):
        pages = [f"第 {i} 页" for i in range(100)]
        for case in CASES.values():
            base = material(pages, case, None).splitlines()
            self.assertNotIn(case["target"], base)
            self.assertTrue(all(base.count(line) == 1 for line in case["distractors"]))
            offsets = []
            for position in ("front", "middle", "end"):
                lines = material(pages, case, POSITIONS[position]).splitlines()
                offsets.append(lines.index(case["target"]))
                lines.remove(case["target"])
                self.assertEqual(lines, base)
            self.assertLess(offsets[0], offsets[1])
            self.assertLess(offsets[1], offsets[2])
