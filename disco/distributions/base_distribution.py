# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from tqdm.autonotebook import trange

from disco.scorers.positive_scorer import Product
from disco.scorers.exponential_scorer import ExponentialScorer
from disco.scorers.boolean_scorer import BooleanScorer
from .distribution import Distribution
from .single_context_distribution import SingleContextDistribution
from disco.samplers.accumulation_sampler import AccumulationSampler
from disco.utils.device import get_device
from disco.utils.helpers import batchify
from disco.utils.moving_average import MovingAverage


class BaseDistribution(Distribution):
    """
    Base distribution class, which can be used
    to build an EBM.
    """

    def constrain(self,
            features, moments=None,
            proposal=None, context_distribution=SingleContextDistribution(''), context_sampling_size=1,
            n_samples=2**9, iterations=1000, learning_rate=0.05, tolerance=1e-5, sampling_size=2**5
        ):
        """
        Constrains features to the base according to their moments,
        so producing an EBM

        Parameters
        ----------
        features: list(feature)
            multiple features to constrain
        moments: list(float)
            moments for the features. There should be as many moments as there are features
        proposal: distribution
            distribution to sample from, if different from self
        context_distribution: distribution
            to contextualize the sampling and scoring
        context_sampling_size:
            size of the batch when sampling context
        n_samples: int
            number of samples to use to fit the coefficients
        learning_rate: float
            multipliers of the delta used when fitting the coefficients
        tolerance: float
            accepted difference between the targets and moments
        sampling_size:
            size of the batch when sampling samples

        Returns
        -------
        exponential scorer with fitted coefficients
        """

        if list != type(features):
            raise TypeError("features should be passed as a list.")

        if not moments:
            return Product(self, *features)

        if list != type(moments):
            raise TypeError("moments should be passed as a list.")
        if not len(features) == len(moments):
            raise TypeError("there should be as many as many moments as there are features.")

        if all([BooleanScorer == type(f) for f in features])\
            and all([1.0 == float(m) for m in moments]):
            return Product(self, *features)

        if not proposal:
            proposal = self

        context_samples, context_log_scores = context_distribution.sample(context_sampling_size)

        proposal_samples = dict()
        proposal_log_scores = dict()
        joint_log_scores = dict()
        feature_scores = dict()
        for (context, log_score) in zip(context_samples, context_log_scores):
            accumulator = AccumulationSampler(proposal, total_size=n_samples)
            proposal_samples[context], proposal_log_scores[context] = accumulator.sample(
                    sampling_size=sampling_size, context=context
                )
            device = get_device(proposal_log_scores[context])
            reference_log_scores = batchify(
                    self.log_score, sampling_size, samples=proposal_samples[context], context=context
                ).to(device)
            joint_log_scores[context] = torch.tensor(log_score).repeat(n_samples).to(device) + reference_log_scores
            feature_scores[context] = torch.stack(
                    ([f.score(proposal_samples[context], context).to(device) for f in features])
                )

        coefficients = torch.tensor(0.0).repeat(len(features)).to(device)
        targets = torch.tensor(moments).to(device)
        with trange(iterations, desc='fitting exponential scorer') as t:
            for i in t:
                scorer = ExponentialScorer(features, coefficients)
                numerator = torch.tensor(0.0).repeat(len(features)).to(device)
                denominator = torch.tensor(0.0).repeat(len(features)).to(device)
                for context in context_samples:
                    target_log_scores = joint_log_scores[context] + scorer.log_score(
                            proposal_samples[context], context
                        ).to(device)
                    importance_ratios = torch.exp(target_log_scores - proposal_log_scores[context])
                    numerator += (importance_ratios * feature_scores[context]).sum(dim=1)
                    denominator += importance_ratios.sum()
                moments = numerator / denominator
                grad_coefficients = moments - targets
                err = grad_coefficients.abs().max().item()
                t.set_postfix(err=err)
                if tolerance > err:
                    t.total_size = i
                    t.refresh()
                    break
                coefficients -= learning_rate * grad_coefficients

        return self * ExponentialScorer(features, coefficients)
