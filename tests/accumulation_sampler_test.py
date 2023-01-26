# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

from disco.distributions import LMDistribution
from disco.samplers import AccumulationSampler

class AccumulationSamplerTest(unittest.TestCase):

    def test_clm_sample(self):
        prefix = "It was a cold and stormy night"
        proposal = LMDistribution()

        total_size = 2**8
        sampler = AccumulationSampler(proposal, total_size=total_size)
        samples, _ = sampler.sample(context=prefix, sampling_size=2**6)

        self.assertEqual(total_size, len(samples),
            "there should as many sampled sequences as requested from the CLM.")


if __name__ == '__main__':
    unittest.main()