# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest
import torch
import numpy as np
import random

from disco.distributions import BaseDistribution
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.scorers import BooleanScorer
from disco.scorers.positive_scorer import PositiveScorer
from disco.scorers.positive_scorer import Product
from disco.scorers.exponential_scorer import ExponentialScorer


class DummyDistribution(BaseDistribution):

    def __init__(self, low=0):
        self.low = low

    def _max(self, high):
        return max(self.low + 8, high)

    def log_score(self, numbers, context=None):
        context = 64 if context is None else context
        high = self._max(context)

        numbers = torch.tensor(numbers)
        probability = 1 / (high - self.low) 
        probabilities = torch.where(
                torch.logical_and(self.low <= numbers, numbers < high), probability, 0.0
            )
        return torch.log(probabilities)

    def sample(self, sampling_size=1, context=None):
        context = 64 if context is None else context
        high = self._max(context)
    
        numbers = torch.randint(self.low, high, (sampling_size,))

        return (
                numbers.tolist(),
                self.log_score(numbers),
            )

even = BooleanScorer(lambda x, _: 0 == x % 2)
divisible_by_3 = BooleanScorer(lambda x, _: 0 == x % 3)
odds = [i * 2 + 1 for i in range(10)]
evens = [i * 2 for i in range(10)]
first_20_integers = odds + evens

class BaseDistributionTest(unittest.TestCase):

    def test_constrain_features_should_passed_as_a_list(self):
        reference = DummyDistribution()
        with self.assertRaises(TypeError) as cm:
            _ = reference.constrain(even)
        err = cm.exception
        self.assertEqual(str(err), "features should be passed as a list.")

    def test_constrain_moments_should_passed_as_a_list_when_any(self):
        reference = DummyDistribution()
        with self.assertRaises(TypeError) as cm:
            _ = reference.constrain([even], 0.5)
        err = cm.exception
        self.assertEqual(str(err), "moments should be passed as a list.")

    def test_constrain_features_should_match_moments(self):
        reference = DummyDistribution()
        with self.assertRaises(TypeError) as cm:
            _ = reference.constrain([even, divisible_by_3], [0.5])
        err = cm.exception
        self.assertEqual(str(err), "there should be as many as many moments as there are features.")

    def test_constrain_pointwisely_with_a_feature(self):
        reference = DummyDistribution()
        target = reference.constrain([even])
        self.assertEqual(Product, type(target),
            "constrain(...) should return a Product.")
        self.assertTrue(all([-np.Inf == s for s in target.log_score(odds, None)]),
            "a base distribution should be constrainable with a single pointwise feature to build an EBM.")
        self.assertTrue(all([-np.Inf < s < 0 for s in target.log_score(evens, None)]),
            "a base distribution should be constrainable with a single pointwise feature to build an EBM.")

    def test_constrain_pointwisely_with_multiple_features(self):
        reference = DummyDistribution()
        target = reference.constrain([even, divisible_by_3])
        self.assertEqual(4, len([s for s in target.log_score(first_20_integers, None) if -np.Inf < s < 0]),
            "a base distribution should be constrainable with multiple pointwise features to build an EBM.")

    def test_constrain_is_like_sugar_when_no_moment(self):
        reference = DummyDistribution()
        target_constrain = reference.constrain([even, divisible_by_3])
        target_sugar = reference * even * divisible_by_3
        some_integers, _ = reference.sample(20, 16)
        self.assertTrue(
            all([c == s for c, s in zip(target_constrain.log_score(some_integers, None), target_sugar.log_score(some_integers, None))]),
            "an EBM built from constrain(…) should be equivalent to the one built with the product sign sugar.")

    def test_constrain_distributionally_with_a_feature(self):
        reference = DummyDistribution()
        target = reference.constrain([even], [0.8], context_distribution=SingleContextDistribution(None))
        self.assertEqual(Product, type(target),
            "constrain(...) should return a Product.")

    def test_constrain_distributionally_with_multiple_features(self):
        reference = DummyDistribution()
        target = reference.constrain([even, divisible_by_3], [0.8, 0.5], context_distribution=SingleContextDistribution(None))
        self.assertEqual(Product, type(target),
            "constrain(...) should return a Product.")

    def test_constrain_without_moments(self):
        bs = BooleanScorer(lambda x, c: True)
        distribution = BaseDistribution(lambda x, c: random.random())
        self.assertEqual(BooleanScorer, type(distribution.constrain([bs]).scorers[-1]),
            "when no moment is specified a feature should be composed directly in a product.")
        for s in distribution.constrain([bs, bs, bs]).scorers[1:]:
            self.assertEqual(BooleanScorer, type(s),
                "when no moments are specified multiple features should appear as is in a product.")

    def test_constrain_with_boolean_scorers_and_moments_set_to_one(self):
        bs = BooleanScorer(lambda x, c: True)
        distribution = BaseDistribution(lambda x, c: 1)
        self.assertEqual(BooleanScorer, type(distribution.constrain([bs], [1]).scorers[-1]),
            "when only a boolean scorer is specified, and its moment is one, this feature should appear as is in a product.")
        bs = BooleanScorer(lambda x, c: True)
        for s in distribution.constrain([bs, bs, bs], [1, 1.0, "1"]).scorers[1:]:
            self.assertEqual(BooleanScorer, type(s),
                "when only boolean scorer are specified, and their moments are one, these feature should appear as is in a product.")

    def test_constrain_with_boolean_scorers_but_with_moments_not_set_to_one_returns_an_exponential_scorer(self):
        bs = BooleanScorer(lambda x, c: True)
        distribution = DummyDistribution()
        self.assertEqual(ExponentialScorer, type(distribution.constrain([bs, bs], [1, 0.5], context_distribution=SingleContextDistribution(None)).scorers[-1]),
            "when only boolean scorers are specified, but their moments are not one, features should be composed in an exponential scorer.")

    def test_constrain_returns_an_exponential_scorer(self):
        bs = BooleanScorer(lambda x, c: True)
        ps = PositiveScorer(lambda x, c: random.random())
        distribution = DummyDistribution()
        self.assertEqual(ExponentialScorer, type(distribution.constrain([bs, ps], [0.2, 0.5], context_distribution=SingleContextDistribution(None)).scorers[-1]),
            "features should be composed by default in an exponential scorer.")

if __name__ == '__main__':
    unittest.main()