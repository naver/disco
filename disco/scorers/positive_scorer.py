# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np
from functools import reduce

from .scorer import Scorer


class PositiveScorer(Scorer):
    """
    Scorer, but limited to positive values
    """

    def __mul__(self, ot):
        """enables the use of the multiplication sign (*)
        to compose positive scorers"""

        return Product(self, ot)

    def log_score(self, samples, context):
        """relies on the instance's scoring function
        to compute the log-scores of the samples given the context

        Parameters
        ----------
        samples : list(Sample)
            list of samples to log-score
        context: text
            context that the samples relate to

        Returns
        -------
        tensor of log-scores for the samples"""

        return torch.log(self.scoring_function(samples, context))

    def score(self, samples, context):
        """returns the scores for the samples
        given the context by exponentiating their log-scores 

        Parameters
        ----------
        samples : list(Sample)
            list of samples to score
        context: text
            context that the samples relate to

        Returns
        -------
        tensor of scores for the samples"""

        return torch.exp(self.log_score(samples, context))


class Product(PositiveScorer):
    """
    Utility class to compose scorers on the product of their scores
    """

    def __init__(self, *scorers):
        self.scorers = scorers

    def log_score(self, samples, context):
        """computes the product of the log-scores,
        hence adds the log-scores from the individual scorers

        Parameters
        ----------
        samples : list(Sample)
            list of samples to log-score
        context: text
            context used for the samples

        Returns:
        --------
            list of log-scores for the samples
        """
        try:
            device = self.scorers[0].device
        except AttributeError:
            device = "cpu"
        
        log_scores = [s.log_score(samples, context).to(device) for s in self.scorers]
        return torch.tensor(reduce(lambda x,y: x+y, log_scores))

    def __str__(self):
        scorers_str = ", ".join((str(scorer) for scorer in self.scorers))
        return f"Product({scorers_str})"
    
