# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

from disco.utils.device import get_device
from .base import BaseDivergence


class TV(BaseDivergence):

    @classmethod
    def pointwise_estimates(cls, m1_log_scores, m2_log_scores, z, proposal_log_scores=None):
        """
        computes the TVD between 2 distributions

        Parameters
        ----------
        m1_log_scores: floats
            log-scores for samples according to network 1
        m2_log_scores: floats
            log-scores for samples according to network 2
        z: float
            partition function of network 1
        proposal_log_scores: floats
            log-scores for samples according to proposal (by default m2_log_scores)

        Returns
        -------
        divergence between m1 and m2
        """

        device = get_device(m1_log_scores)

        if isinstance(z, float):
            z = torch.tensor(z, device=device, dtype=m1_log_scores.dtype)

        m2_log_scores = m2_log_scores.to(device)

        if proposal_log_scores is None:
            proposal_log_scores = m2_log_scores
        else:
            proposal_log_scores = proposal_log_scores.to(device)

        normalized_m1_log_scores  = m1_log_scores - torch.log(z)

        return 1/2 * (torch.abs(torch.exp(m2_log_scores - proposal_log_scores) -
                torch.exp(normalized_m1_log_scores - proposal_log_scores)))
