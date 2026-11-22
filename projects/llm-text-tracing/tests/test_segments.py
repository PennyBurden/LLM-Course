import unittest

import numpy as np

from tracing.segments import CHANGE, COUNT, MEAN, WIDTH, adjacent_distances, segment_ranges, segment_vectors


class SegmentTest(unittest.TestCase):
    def test_ranges_keep_the_ends_and_never_repeat_a_token(self):
        for length in [*range(1, 161), 4096]:
            with self.subTest(length=length):
                ranges = segment_ranges(length)
                positions = [i for lo, hi in ranges for i in range(lo, hi)]
                self.assertEqual(len(ranges), COUNT)
                self.assertTrue(all(0 <= hi - lo <= WIDTH for lo, hi in ranges))
                self.assertEqual((positions[0], positions[-1]), (0, length - 1))
                self.assertEqual(positions, sorted(set(positions)))
                self.assertEqual(len(positions), min(length, COUNT * WIDTH))
        with self.assertRaises(ValueError):
            segment_ranges(0)

    def test_empty_slots_do_not_introduce_a_transition(self):
        slots = np.zeros((2, COUNT, 2), dtype=np.float32)
        slots[:, 0] = [3, 0]
        slots[1, 1] = [0, 5]
        values = segment_vectors(slots, CHANGE)
        np.testing.assert_allclose(values[0], [1, 0, 0, 0])
        np.testing.assert_allclose(values[1], [0.5] * 4, atol=1e-6)
        self.assertEqual(adjacent_distances(slots[0]), [])
        self.assertEqual(adjacent_distances(slots[1]), [1])

    def test_mean_is_order_invariant_but_adjacent_changes_keep_order(self):
        slots = np.array([[[1, 0], [1, 0], [0, 1], [-1, 0]]], dtype=np.float32)
        permuted = slots[:, [0, 3, 1, 2]]
        np.testing.assert_allclose(segment_vectors(slots, MEAN), segment_vectors(permuted, MEAN))
        self.assertFalse(np.allclose(segment_vectors(slots, CHANGE), segment_vectors(permuted, CHANGE)))
        self.assertEqual(adjacent_distances(slots[0]), [0, 1, 1])
