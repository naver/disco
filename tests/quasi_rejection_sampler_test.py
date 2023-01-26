# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

from disco.distributions import LMDistribution
from disco.scorers import BooleanScorer
from disco.samplers import QuasiRejectionSampler

class QuasiRejectionSamplerTest(unittest.TestCase):

    def test_sample(self):
        prefix = "It was a cold and stormy"
        word = "night"
        target = LMDistribution() * BooleanScorer(lambda s, c: word in s.text)
        proposal = LMDistribution()

        sampler = QuasiRejectionSampler(target, proposal)
        samples, _ = sampler.sample(sampling_size=32, context=prefix)

        self.assertTrue(all(word in s.text for s in samples),
            "sampled sequences should respect the constraints.")

        self.assertGreater(sampler.get_acceptance_rate(), 0.,
            "that sampling should probably return at least a sample.")


if __name__ == '__main__':
    unittest.main()