# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from .f_divergence import FDivergenceLoss

class KLLoss(FDivergenceLoss):
    """
    Kullback-Leibler divergence loss for DPG
    """
    def __init__(self, use_baseline=True, baseline_window_size=1024):
        """
        Parameters
        ----------
        use_baseline: boolean
            use a baseline to reduce variance
        """
        super(KLLoss, self).__init__(use_baseline, baseline_window_size)

    def f_prime(self, log_t):
        """
        Parameters
        ----------
        log_t: 0-dim Tensor
            The log ratio of the policy and the normalized target distribution
        """
        return -torch.exp(-log_t)