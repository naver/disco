# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

from disco.utils.device import get_device

class BaseDivergence:
    """
    Kullback-Leibler divergence class.
    """

    @classmethod
    def divergence(cls, m1_log_scores, m2_log_scores, z, proposal_log_scores=None):
        """
        Computes an IS of the KL divergence between 2 distributions

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

        return torch.mean(cls.pointwise_estimates(
            m1_log_scores, m2_log_scores, z, proposal_log_scores=proposal_log_scores))
