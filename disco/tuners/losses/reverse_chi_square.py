# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from .f_divergence import FDivergenceLoss

class ReverseChiSquaredLoss(FDivergenceLoss):
    """
    Reverse Chi Squared divergence χ²(π || p) loss for DPG
    """
    def __init__(self, use_baseline=True, baseline_window_size=1024):
        """
        Parameters
        ----------
        use_baseline: boolean
            use a baseline to reduce variance
        """
        super(ReverseChiSquaredLoss, self).__init__(use_baseline, baseline_window_size)

    def f_prime(self, log_t):
        """
        Computes the f' term for the reverse (Neyman) chi-square divergence.

        Parameters
        ----------
        log_t: 0-dim Tensor
            The log ratio of the policy and the normalized target distribution, log(p/q).
        """
        return (torch.exp(log_t) - 1) * 2