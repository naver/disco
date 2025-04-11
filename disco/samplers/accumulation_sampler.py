# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import Sampler
import torch
from tqdm.autonotebook import trange


class AccumulationSampler(Sampler):
    """
    Utility class to accumulate samples, up to a total size
    """

    def __init__(self, distribution, total_size=512):
        """
        Parameters
        ----------
        distribution: distribution
            distribution to sample from
        total_size: int
            total number of samples
        """

        self.distribution = distribution
        self.total_size = total_size

    def sample(self, sampling_size=32, context=""):
        """accumulates batches of samples from the distribution

        Parameters
        ----------
        sampling_size: int
            number of requested samples per individual sampling
        context: text
            contextual text for which to sample

        Returns
        -------
        a tuple of accumulated samples and scores
        """
        with trange(
                self.total_size,
                desc=f"sampling from {type(self.distribution).__name__}",
                position=1,
                leave=False
            ) as t:
            remaining = self.total_size
            samples, log_scores = list(), torch.empty([0])
            while remaining > 0:
                more_samples, more_log_scores = self.distribution.sample(context=context, sampling_size=sampling_size)
                length = min(remaining, len(more_samples))
                more_samples, more_log_scores = more_samples[:length], more_log_scores[:length]
                samples, log_scores = (
                        samples + more_samples,
                        torch.cat((log_scores, more_log_scores))
                    ) if samples else (more_samples, more_log_scores)
                remaining -= len(more_samples)
                t.update(len(more_samples))

        return (samples, log_scores)