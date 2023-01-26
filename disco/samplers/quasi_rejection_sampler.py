# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

from . import Sampler
from disco.utils.device import get_device


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
        """Computes the acceptance rate, that is the number of accepted samples
        over the total sampled ones
        
        Returns
        -------
        acceptance rate as float between 0 and 1"""

        return self.n_accepted_samples / self.n_samples
