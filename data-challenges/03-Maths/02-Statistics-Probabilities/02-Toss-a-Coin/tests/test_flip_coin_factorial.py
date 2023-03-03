
import unittest
from flip_coin_factorial import count_possibilities, count_total_possibilities, probability

class TestTheory(unittest.TestCase):

    # Test for count_possibilities
    def test_count_possibilities_11(self):
        expected = 1
        self.assertEqual(count_possibilities(1, 1), expected)

    def test_count_possibilities_44(self):
        expected = 1
        self.assertEqual(count_possibilities(4, 4), expected)

    def test_count_possibilities_42(self):
        expected = 6
        self.assertEqual(count_possibilities(4, 2), expected)

    def test_count_possibilities_43(self):
        expected = 4
        self.assertEqual(count_possibilities(4, 3), expected)


    # Test for count_total_possibilities
    def test_count_total_possibilities_1(self):
        expected = 2
        self.assertEqual(count_total_possibilities(1), expected)

    def test_count_total_possibilities_2(self):
        expected = 16
        self.assertEqual(count_total_possibilities(4), expected)

    def test_count_total_possibilities_6(self):
        expected = 64
        self.assertEqual(count_total_possibilities(6), expected)

    def test_count_total_possibilities_10(self):
        expected = 1024
        self.assertEqual(count_total_possibilities(10), expected)


    # Test for probability
    def test_probability_1(self):
        expected = {0: 0.5, 1: 0.5}
        self.assertEqual(probability(1), expected)

    def test_probability_4(self):
        expected = {0: 0.0625, 1: 0.25, 2: 0.375, 3: 0.25, 4: 0.0625}
        self.assertEqual(probability(4), expected)

    def test_probability_100(self):
        expected = 0.08
        self.assertAlmostEqual(probability(100)[50], expected, places=2)


