# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from disco.scorers import ExponentialScorer
from disco.scorers import BooleanScorer


rain = lambda s, c: "rain" in s.text
city = lambda s, c: "city" in s.text

class ExponentialScorerTest(unittest.TestCase):

    def test_features_and_coefficients_match(self):
        scorer = ExponentialScorer([BooleanScorer(rain), BooleanScorer(city)], [0.5, 0.25])
        self.assertTrue(hasattr(scorer, "features"),
            "the exponential scorer should have a features attribute.")
        self.assertTrue(hasattr(scorer, "coefficients"),
            "the exponential scorer should have a coefficients attribute.")
        self.assertEqual(len(scorer.features), len(scorer.coefficients),
            "the length of both features and coefficients list should be equal.")

    def test_features_and_coefficients_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            ExponentialScorer(
                    [BooleanScorer(rain)],
                    [0.5, 0.25]
                )

    def test_coefficients_as_tensor_like(self):
        with self.assertRaises(TypeError) as cm:
            ExponentialScorer(
                    [BooleanScorer(rain)],
                    0.5
                )
        with self.assertRaises(TypeError) as cm:
            ExponentialScorer(
                    [BooleanScorer(rain), BooleanScorer(city)],
                    {"rain": 0.5, "city": 0.25}
                )

    def test_score(self):
        scorer = ExponentialScorer([BooleanScorer(rain), BooleanScorer(city)], [0.5, 0.25])
        texts = [
                "I'm singing in the rain.",
                "What is the city but the people?",
                "The rain that fell on the city runs down the dark gutters and empties into the sea without even soaking the ground"
                "Every drop in the ocean counts."
            ]

        from disco.distributions.lm_distribution import TextSample
        samples = [TextSample(list(), t) for t in texts] # fake samples without the tokenizations

        scores = scorer.score(samples, None)
        self.assertEqual(len(samples), len(scores),
            "there should be a score for each sample.")
        log_scores = scorer.log_score(samples, None)
        self.assertEqual(len(samples), len(log_scores),
            "there should be a (log-)score for each sample.")
        for e, s in zip([0.5, 0.25, 0.75, 0], log_scores):
            self.assertEqual(e, s,
                "the exponential scorer should (log-)score correctly."
            )


if __name__ == '__main__':
    unittest.main()