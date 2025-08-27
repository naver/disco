# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .f_divergence import FDivergenceLoss
import torch

class JSLoss(FDivergenceLoss):
    """
    Jensen-Shannon divergence loss for DPG
    """
    def __init__(self, use_baseline=True, baseline_window_size=1024):
        """
        Parameters
        ----------
        use_baseline: boolean
            use a baseline to reduce variance
        """
        super(JSLoss, self).__init__(use_baseline, baseline_window_size)

    def f_prime(self, log_t):
        """
        Parameters
        ----------
        log_t: 0-dim Tensor
            The log ratio of the policy and the normalized target distribution
        """
        log_2 = torch.log(torch.tensor(2.0, dtype=log_t.dtype, device=log_t.device))
        return log_2 - torch.nn.functional.softplus(-log_t)