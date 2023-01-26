# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

import numpy as np

from disco.scorers import PipelineScorer

class PipelineScorerTest(unittest.TestCase):

    def test_detection(self):
        params = {
            "task": "sentiment-analysis",
            "model": "siebert/sentiment-roberta-large-english"
        }
        pf = PipelineScorer('POSITIVE', params)

        texts = [
                "This is so interesting: I love it!",
                "How can you be so depressing? I don't want to go there anymore."
            ]

        from disco.distributions.lm_distribution import TextSample
        samples = [TextSample(list(), t) for t in texts] # fake samples without the tokenizations

        log_scores = pf.log_score(samples, None)
        expected_sentiment = [1, -1]
        self.assertTrue(all(e == np.sign(np.exp(s)-0.5) for e, s in zip(expected_sentiment, log_scores)),
            "a sentiment should be correctly classified.")


if __name__ == '__main__':
    unittest.main()
