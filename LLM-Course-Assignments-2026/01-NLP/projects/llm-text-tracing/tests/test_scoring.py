import unittest

from tracing.data import read_rows
from tracing.metrics import evaluate


class ScoreTest(unittest.TestCase):
    def test_missed_machine_is_a_family_false_negative(self):
        gold = {i: {"id": i, "label": 1, "family": i} for i in range(8)}
        gold[8] = {"id": 8, "label": 0}
        gold[9] = {"id": 9, "label": 2}
        predicted = {i: {"family": -1, **row} for i, row in reversed(list(gold.items()))}
        perfect = evaluate(gold, predicted)
        self.assertEqual(perfect["joint_score"], 1.0)
        predicted[4] = {"id": 4, "label": 0, "family": -1}
        result = evaluate(gold, predicted)
        self.assertEqual(result["source_macro_f1"], 7 / 8)
        self.assertEqual(result["machine_missed"], 1)

    def test_bad_ids_and_inconsistent_family_are_rejected(self):
        gold = {1: {"id": 1, "label": 0}}
        with self.assertRaises(ValueError):
            evaluate(gold, {})
        with self.assertRaises(ValueError):
            evaluate(gold, {1: {"id": 1, "label": 0, "family": 2}})
        with self.assertRaises(ValueError):
            read_rows(['{"id": 1}', '{"id": 1}'])
        with self.assertRaises(ValueError):
            read_rows(['{"id": true}'])

    def test_false_alarm_on_human_only_affects_detection(self):
        gold = {i: {"id": i, "label": 1, "family": i} for i in range(8)}
        gold[8] = {"id": 8, "label": 0}
        gold[9] = {"id": 9, "label": 2}
        predicted = {i: {"family": -1, **row} for i, row in gold.items()}
        predicted[8] = {"id": 8, "label": 1, "family": 4}
        result = evaluate(gold, predicted)
        self.assertEqual(result["source_macro_f1"], 1)
        self.assertEqual(result["n_machine"], 8)
        self.assertLess(result["detect_macro_f1"], 1)


if __name__ == "__main__":
    unittest.main()
