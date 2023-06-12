# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

from . import Sampler
from .accumulation_sampler import AccumulationSampler
from disco.metrics import KL
from disco.utils.device import get_device
from disco.utils.helpers import batchify


class QuasiRejectionSampler(Sampler):
    """
    Quasi Rejection-Sampling class
    """

    def __init__(self, target, proposal, beta=1):
        """
        Parameters
        ----------
        target: distribution
            Energy-based model to (log-)score the samples
        proposal: distribution
            distribution to generate the samples
        beta: float
            coefficient to control the sampling
        """

        super(QuasiRejectionSampler, self).__init__(target, proposal)
        self.beta = beta
        self.n_samples = 0
        self.n_accepted_samples = 0

    def sample(self, sampling_size=32, context=''):
        """Generates samples according to the QRS algorithm

        Parameters
        ----------
        sampling_size: int
            number of requested samples when sampling
        context: text
            contextual text for which to sample

        Returns
        -------
        tuple of accepted samples and their log-scores
        """

        samples, proposal_log_scores = self.proposal.sample(sampling_size=sampling_size, context=context)
        self.n_samples += len(samples)

        device = get_device(proposal_log_scores)

        target_log_scores = self.target.log_score(samples=samples, context=context).to(device)

        rs = torch.clamp(
                torch.exp(target_log_scores - proposal_log_scores) / self.beta,
                min=0.0, max=1.0
            )

        us = torch.rand(len(rs)).to(device)
        accepted_samples = [x for k, x in zip(us < rs, samples) if k]
        self.n_accepted_samples += len(accepted_samples)
        accepted_log_scores = torch.tensor([s for k, s in zip(us < rs, proposal_log_scores) if k]).to(device)

        return accepted_samples, accepted_log_scores

    def get_acceptance_rate(self):
        """Computes the currently observed acceptance rate, that is the number
        of accepted samples over the total sampled ones

        Returns
        -------
        acceptance rate as float between 0 and 1"""

        return self.n_accepted_samples / self.n_samples


class QuasiRejectionSamplerEstimator:
    """
    Provides routines to compute estimates of useful metrics related to
    QuasiRejectionSampler (QRS)
    """

    def __init__(self, target, proposal, n_estimation_samples=10000,
            sampling_size=32, context=''):
        """
        Parameters
        ----------
        target: distribution
            Energy-based model to (log-)score the samples
        proposal: distribution
            distribution to generate the samples
        n_estimation_samples: integer
            number of samples to use for computing estimates
        sampling_size: integer
            number of samples that are concurrently obtained from the proposal
        context: text
            context to condition the proposal and target
        """
        sampler = AccumulationSampler(proposal, total_size=n_estimation_samples)

        self.samples, self.proposal_log_scores = \
                sampler.sample(sampling_size=sampling_size, context=context)

        self.target_log_scores = batchify(target.log_score,
                sampling_size, samples=self.samples, context=context)

    def acceptance_rate_at_beta(self, beta):
        """
        Estimate the acceptance rate that QRS has for a given beta parameter.

        Parameters
        ----------
        beta: float
            the value of beta for which we want the a.r. estimated

        Returns
        -------
        the estimated acceptance rate for the given beta
        """
        vals = self._compute_intermediary_values(beta)

        return vals['qrs_Z'].item() / beta

    def divergence_at_beta(self, beta, divergence=KL):
        """
        Estimate the divergence to the target distribution that QRS has
        for a given beta parameter.

        Parameters
        ----------
        beta: float
            the value of beta for which we want the a.r. estimated
        divergence: divergence
            the divergence that we want to compute (see :py:mod:`metrics`)

        Returns
        -------
        the estimated value of the chosen divergence for the given beta
        """
        vals = self._compute_intermediary_values(beta)

        return divergence.divergence(
                vals['target_log_scores'],
                vals['normalized_qrs_log_scores'],
                vals['Z'],
                vals['proposal_log_scores']).item()

    def feature_moment_at_beta(self, beta, feature):
        """
        Estimate the first moment (expected value) of a feature when using
        QRS with a given beta parameter.

        Parameters
        ----------
        beta: float
            the value of beta for which we want the a.r. estimated
        feature: scorer
            the feature whose moment we want to compute

        Returns
        -------
        the estimated feature moment at the given beta
        """
        vals = self._compute_intermediary_values(beta)

        feature_scores = batchify(feature.score,
                sampling_size, samples=self.estimation_samples, context=context)

        return torch.mean(vals['qrs_importance_ratios'] * feature_scores).item()

    def _compute_intermediary_values(self, beta):
        """
        Computes intermediary values used in QRS estimates
        """
        with torch.no_grad():
            target_importance_ratios = torch.exp(self.target_log_scores -
                    self.proposal_log_scores)
            Z = torch.mean(target_importance_ratios)
            log_beta = torch.log(torch.tensor(beta)) if beta > 0 else float("-inf")
            qrs_log_scores = torch.minimum(self.target_log_scores,
                    self.proposal_log_scores + log_beta)
            qrs_Z = torch.mean(torch.exp(qrs_log_scores - self.proposal_log_scores))
            normalized_qrs_log_scores = qrs_log_scores - torch.log(qrs_Z)
            qrs_importance_ratios = torch.exp(normalized_qrs_log_scores -
                    self.proposal_log_scores)

        return  {
                'target_log_scores': self.target_log_scores,
                'proposal_log_scores': self.proposal_log_scores,
                'qrs_log_scores': qrs_log_scores,
                'normalized_qrs_log_scores': normalized_qrs_log_scores,
                'target_importance_ratios': target_importance_ratios,
                'Z': Z,
                'qrs_Z': qrs_Z,
                'qrs_importance_ratios': qrs_importance_ratios}
