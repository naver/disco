# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

from .distribution import Distribution

class SingleContextDistribution(Distribution):
    """
    Single context distribution class, useful to sample the
    same context that is to fall back to a fixed-context case.
    """

    def __init__(self, context=''):
        """
        Parameters
        ----------
        context: string
            unique context to return when sampling
        """

        self.context = context

    def log_score(self, contexts):
        """Computes log-probabilities of the contexts
        to match the instance's context

        Parameters
        ----------
        contexts: list(str)
            list of contexts to (log-)score

        Returns
        -------
        tensor of log-probabilities
        """

        return torch.tensor([0 if self.context == context else -float("inf") for context in contexts])

    def sample(self, sampling_size=32):
        """Samples multiple copies of the instance's context
        
        Parameters
        ----------
        sampling_size: int
            number of contexts to sample
        
        Returns
        -------
        tuple of (list of texts, tensor of log-probabilities)
        """
    
        return (
                [self.context] * sampling_size,
                [0] * sampling_size
            )