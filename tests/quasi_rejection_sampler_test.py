# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

from disco.distributions import LMDistribution
from disco.scorers import BooleanScorer
from disco.samplers import QuasiRejectionSampler
from disco.samplers.quasi_rejection_sampler import QuasiRejectionSamplerEstimator
from disco.metrics import KL, TV
from scipy import stats
import torch

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

    def test_estimator(self):
        proposal = PoissonDistribution(10)
        target = PoissonDistribution(11)
        estimator = QuasiRejectionSamplerEstimator(target, proposal, n_estimation_samples=int(10e4))
        ar = estimator.acceptance_rate_at_beta(0.5)
        self.assertAlmostEqual(ar, 0.999, places=2)

        ar = estimator.acceptance_rate_at_beta(1)
        self.assertAlmostEqual(ar, 0.877, places=2)

        ar = estimator.acceptance_rate_at_beta(2)
        self.assertAlmostEqual(ar, 0.5, places=2)

        tvd = estimator.divergence_at_beta(0.5, divergence=TV)
        self.assertAlmostEqual(tvd, 0.12, places=2)

        tvd = estimator.divergence_at_beta(1, divergence=TV)
        self.assertAlmostEqual(tvd, 0.075, places=2)

        tvd = estimator.divergence_at_beta(2, divergence=TV)
        self.assertAlmostEqual(tvd, 0.0041, places=2)

        kl = estimator.divergence_at_beta(1, divergence=KL)
        self.assertAlmostEqual(kl, 0.021, places=2)

        kl = estimator.divergence_at_beta(0.5, divergence=KL)
        self.assertAlmostEqual(kl, 0.046, places=2)

        kl = estimator.divergence_at_beta(2, divergence=KL)
        self.assertAlmostEqual(kl, 0.0, places=2)

class PoissonDistribution(object):

    def __init__(self, lam):
        self.lam = lam

    def sample(self, sampling_size=1, context=None):
        samples = list(stats.poisson.rvs(self.lam, size=sampling_size))
        return samples, self.log_score(samples)

    def log_score(self, x, context=None):
        return torch.tensor(stats.poisson.logpmf(x, self.lam))

if __name__ == '__main__':
    unittest.main()
