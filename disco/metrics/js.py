# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import logging
from disco.utils.device import get_device
from .base import BaseDivergence
from .kl import KL


logger = logging.getLogger(__name__)

class JS(BaseDivergence):
    """
    Jensen-Shannon divergence class.
    """

    @classmethod
    def pointwise_estimates(cls, m1_log_scores, m2_log_scores, z, proposal_log_scores=None):
        """
        Computes the KL divergence between 2 distributions

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

        m2_log_scores = m2_log_scores.to(device)
        normalized_m1_log_scores = m1_log_scores - torch.log(z)

        max_log_scores = torch.max(normalized_m1_log_scores, m2_log_scores)

        m_log_scores = max_log_scores + torch.log(((normalized_m1_log_scores - max_log_scores).double().exp() + (m2_log_scores - max_log_scores).double().exp()) / 2).float()

        divergence = KL.pointwise_estimates(normalized_m1_log_scores, m_log_scores, torch.as_tensor(1), proposal_log_scores) / 2 + \
                KL.pointwise_estimates(m2_log_scores, m_log_scores, torch.as_tensor(1), proposal_log_scores) / 2
        return divergence
