# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from disco.scorers import Product
from disco.scorers import BooleanScorer

class ProductFeatureTest(unittest.TestCase):

    def test_single_feature(self):
        integers = np.random.random_integers(0, 5, 100)
        bf = BooleanScorer(lambda x, _: 5 < x)
        bf_log_scores = bf.log_score(integers, None)
        product = Product(bf)
        pr_log_scores = product.log_score(integers, None)
        self.assertTrue(all(b == p for b, p in zip(bf_log_scores, pr_log_scores)),
            "a product of a single feature should behave like that feature.")

    def test_two_features(self):
        integers = np.random.random_integers(0, 5, 100)
        bf1 = BooleanScorer(lambda x, _: 5 < x)
        bf1_log_scores = bf1.log_score(integers, None)
        bf2 = BooleanScorer(lambda x, _: 8 > x)
        bf2_log_scores = bf2.log_score(integers, None)
        product = Product(bf1, bf2)
        pr_log_scores = product.log_score(integers, None)
        self.assertTrue(all((b1 + b2) == p for b1, b2, p in zip(bf1_log_scores, bf2_log_scores, pr_log_scores)),
            "a product of two features should sum their respective log-scores.")

    def test_three_features(self):
        integers = np.random.random_integers(0, 5, 100)
        bf1 = BooleanScorer(lambda x, _: 2 < x)
        bf1_log_scores = bf1.log_score(integers, None)
        bf2 = BooleanScorer(lambda x, _: 8 > x)
        bf2_log_scores = bf2.log_score(integers, None)
        bf3 = BooleanScorer(lambda x, _: 0 == x % 2)
        bf3_log_scores = bf3.log_score(integers, None)
        product = Product(bf1, bf2, bf3)
        pr_log_scores = product.log_score(integers, None)
        self.assertTrue(all((b1 + b2 + b3) == p for b1, b2, b3, p in zip(bf1_log_scores, bf2_log_scores, bf3_log_scores, pr_log_scores)),
            "a product of three features should sum their respective log-scores.")

    def test_product_is_commutative(self):
        integers = np.random.random_integers(0, 5, 100)
        bf1 = BooleanScorer(lambda x, _: 5 < x)
        bf2 = BooleanScorer(lambda x, _: 8 > x)
        pr12 = Product(bf1, bf2)
        pr21 = Product(bf1, bf2)
        self.assertTrue(all(p12 == p21 for p12, p21 in zip(pr12.log_score(integers, None), pr21.log_score(integers, None))),
            "a product of two features should be commutative.")

    def test_product_has_a_sugar(self):
        integers = np.random.random_integers(0, 5, 100)
        bf1 = BooleanScorer(lambda x, _: 5 < x)
        bf1_log_scores = bf1.log_score(integers, None)
        bf2 = BooleanScorer(lambda x, _: 8 > x)
        bf2_log_scores = bf2.log_score(integers, None)
        product = bf1 * bf2
        pr_log_scores = product.log_score(integers, None)
        self.assertTrue(all((b1 + b2) == p for b1, b2, p in zip(bf1_log_scores, bf2_log_scores, pr_log_scores)),
            "a product also works when defined through its syntactic sugar.")


if __name__ == '__main__':
    unittest.main()