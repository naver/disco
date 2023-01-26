# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np


class Scorer():
    """
    Generic scorer
    """

    def __init__(self, scoring_function):
        """
        Parameters
        ----------
        scoring_function: scoring function
            function to be used on each sample
        """

        self.scoring_function = self._broadcast(scoring_function)

    def _broadcast(self, function):
        def broadcasted_function(xs, context):
            return torch.tensor(
                    np.array([function(x, context) for x in xs])
                )
        return broadcasted_function

    def score(self, samples, context):
        """Relies on the instance's scoring function to compute
        scores for the samples given the context

        Parameters
        ----------
        samples : list()
            the samples to score, as a list
        context: text
            context that the samples relate to

        Returns
        -------
        tensor of scores for the samples"""

        return self.scoring_function(samples, context)