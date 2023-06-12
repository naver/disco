# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest

from disco.distributions import LMDistribution
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.scorers import ExponentialScorer, Scorer, BooleanScorer
from disco.tuners import Tuner, DPGTuner, FDPGTuner, CDPGTuner, FCDPGTuner
from disco.tuners.losses import *
from disco.tuners.loggers.base import BaseTunerObserver

class TunerTest(unittest.TestCase):

    def test_rkl(self):
        self._test_loss_rlhf(ReverseKLLoss)

    def test_tv(self):
        self._test_loss_rlhf(TVLoss)

    def test_js(self):
        self._test_loss_rlhf(JSLoss)

    def test_kl(self):
        self._test_loss_rlhf(KLLoss)

    def _test_loss_rlhf(self, loss_cls):
        base = LMDistribution()
        model = LMDistribution(freeze=False)
        reward = Scorer(lambda s, c: 1 if "amazing" in s.text else 0)
        rlhf_target = base * ExponentialScorer([reward], [0.1])
        loss = loss_cls()
        tuner = Tuner(model, rlhf_target, loss=loss, n_gradient_steps=1,
                context_sampling_size=1,
                n_samples_per_step=32,
                scoring_size=32)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('loss' in obs.observations)

    def test_dpg(self):
        base = LMDistribution()
        scorer = BooleanScorer(lambda s, c: "amazing" in s.text)
        target = base.constrain([scorer], [0.5])
        model = LMDistribution(freeze=False)
        tuner = DPGTuner(model, target, n_gradient_steps=1,
                n_samples_per_step=32,
                scoring_size=32)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('loss' in obs.observations)

    def test_cdpg(self):
        base = LMDistribution()
        scorer = BooleanScorer(lambda s, c: "amazing" in s.text)
        target = base.constrain([scorer], [0.5])
        model = LMDistribution(freeze=False)
        context_distribution = SingleContextDistribution("Speaking about today's dinner, it was")
        tuner = CDPGTuner(model, target,
                context_distribution=context_distribution,
                n_gradient_steps=1,
                context_sampling_size=1,
                n_samples_per_step=1,
                scoring_size=1)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('loss' in obs.observations)

    def test_fdpg(self):
        base = LMDistribution()
        scorer = BooleanScorer(lambda s, c: "amazing" in s.text)
        target = base.constrain([scorer], [0.5])
        model = LMDistribution(freeze=False)
        tuner = FDPGTuner(model, target, n_gradient_steps=1,
                n_samples_per_step=1,
                scoring_size=1)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('loss' in obs.observations)

    def test_fcdpg(self):
        base = LMDistribution()
        scorer = BooleanScorer(lambda s, c: "amazing" in s.text)
        target = base.constrain([scorer], [0.5])
        model = LMDistribution(freeze=False)
        context_distribution = SingleContextDistribution("Speaking about today's dinner, it was")
        tuner = FCDPGTuner(model, target,
                context_distribution=context_distribution,
                n_gradient_steps=1,
                context_sampling_size=1,
                n_samples_per_step=1,
                scoring_size=1)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('loss' in obs.observations)

    def test_moments(self):
        base = LMDistribution()
        scorer = BooleanScorer(lambda s, c: "amazing" in s.text)
        target = base.constrain([scorer], [1])
        model = LMDistribution(freeze=False)
        context_distribution = SingleContextDistribution("The movie was absolutely")
        tuner = FCDPGTuner(model, target,
                context_distribution=context_distribution,
                n_gradient_steps=1,
                context_sampling_size=1,
                n_samples_per_step=128,
                divergence_evaluation_interval=1,
                scoring_size=32,
                features=[('amazing', scorer)])
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('amazing_proposal' in obs.observations)
            self.assertTrue('amazing_target' in obs.observations)
            self.assertAlmostEqual(obs.observations['amazing_proposal'].item(), 0.1, 1)
            self.assertAlmostEqual(obs.observations['amazing_target'].item(), 0.1, 1)
            self.assertAlmostEqual(obs.observations['amazing_proposal'].item(), 
                    obs.observations['amazing_target'].item(), 4)
        tuner = FCDPGTuner(model, target,
                context_distribution=context_distribution,
                n_gradient_steps=1,
                context_sampling_size=1,
                n_samples_per_step=1,
                scoring_size=1)
        with MockObserver(tuner) as obs:
            tuner.tune()
            self.assertTrue('amazing_proposal' not in obs.observations)
            self.assertTrue('amazing_target' not in obs.observations)

class MockObserver(BaseTunerObserver):
    def __init__(self, tuner):
        super(MockObserver, self).__init__(tuner)
        self.observations = {}
    def on_metric_updated(self, k, v):
        self.observations[k] = v
