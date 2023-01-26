# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from disco.scorers import BooleanScorer

class BooleanScorerTest(unittest.TestCase):

    def test_truism(self):
        bf = BooleanScorer(lambda x, _: True)
        items = [1, 1.2345, "blah"]
        self.assertEqual(len(items), len(bf.score(items, None)),
            "a boolean feature should score all items.")
        self.assertTrue(all(1. == s for s in bf.score(items, None)),
            "a truism should always score 1.")
        self.assertEqual(len(items), len(bf.log_score(items, None)),
            "a boolean feature should (log-)score all items.")
        self.assertTrue(all(0 == s for s in bf.log_score(items, None)),
            "a truism should always (log-)score 0.")

    def test_true_and_false(self):
        bf = BooleanScorer(lambda x, _: 2 < len(x))
        scores = bf.score(["a", "abc"], None)
        self.assertEqual(0, scores[0],
            "False should result in 0.")
        self.assertEqual(1, scores[1],
            "True should result in 1.")
        log_scores = bf.log_score(["a", "abc"], None)
        self.assertEqual(-np.Inf, log_scores[0],
            "False should result in an infinite negative in log space.")
        self.assertEqual(0, log_scores[1],
            "True should result in a zero in log space.")

    def test_lambda_predicate(self):
        bf = BooleanScorer(lambda x, _: 2 < len(x))
        samples = ["", "a", "ab", "abc", "abcd"]
        scores = bf.score(samples, None)
        expected_scores = [0, 0, 0, 1, 1]
        self.assertTrue(all(e == s for e, s in zip(expected_scores, scores)),
            "a predicate expressed via a lambda should score correctly the items.")
        log_scores = bf.log_score(samples, None)
        expected_log_scores = [-np.Inf, -np.Inf, -np.Inf, 0, 0]
        self.assertTrue(all(e == s for e, s in zip(expected_log_scores, log_scores)),
            "a predicate expressed via a lambda should (log-)score correctly the items.")

    def test_fn_predicate(self):
        def longer_than_2(x, _):
            return 2 < len(x)
        bf = BooleanScorer(longer_than_2)
        samples = ["", "a", "ab", "abc", "abcd"]
        scores = bf.score(samples, None)
        expected_scores = [0, 0, 0, 1, 1]
        self.assertTrue(all(e == s for e, s in zip(expected_scores, scores)),
            "a predicate expressed via a regular function should score correctly the items.")
        log_scores = bf.log_score(samples, None)
        expected_log_scores = [-np.Inf, -np.Inf, -np.Inf, 0, 0]
        self.assertTrue(all(e == s for e, s in zip(expected_log_scores, log_scores)),
            "a predicate expressed via a regular function should (log-)score correctly the items.")

if __name__ == '__main__':
    unittest.main()