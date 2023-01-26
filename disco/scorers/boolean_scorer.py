# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np

from .positive_scorer import PositiveScorer


class BooleanScorer(PositiveScorer):
    """
    Predicate-based scoring class
    """

    def __init__(self, predicate):
        """
        Parameters
        ----------
        predicate: scoring predicate
            predicate function to be used on each sample
        """

        self.predicate = self._broadcast(predicate)

    def log_score(self, samples, context):
        """Returns log-probabilities for the samples
        given the context by converting their scores to logspace 

        Parameters
        ----------
        samples : list(Sample)
            samples to score, as a list
        context: text
            context used for the samples

        Returns
        -------
        tensor of (-np.Inf / 0) log-probabilities"""

        return torch.log(self.score(samples, context))

    def score(self, samples, context):
        """Computes probabilities for samples and context
        by casting the instance's predicate, ie scoring, function

        Parameters
        ----------
        samples : list(Sample)
            samples to score, as a list
        context: text
            context used for the samples

        Returns
        -------
        tensor of (0 / 1) probabilities"""

        return self.predicate(samples, context).float()