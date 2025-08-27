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

        It corresponds to the generator f(u) = (u-1)^2 / u,
        whose derivative is f'(u) = 1 - 1/u^2.

        Parameters
        ----------
        log_t: 0-dim Tensor
            The log ratio of the policy and the normalized target distribution, log(p/q).
        """
        return 1.0 - torch.exp(-2 * log_t)