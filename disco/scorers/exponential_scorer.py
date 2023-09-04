# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np

from .positive_scorer import PositiveScorer
from disco.utils.device import get_device


class ExponentialScorer(PositiveScorer):
    """Exponential scorer to add distributional constraints
    when building an EBM.
    """

    def __init__(self, features, coefficients):
        """
        Parameters
        ----------
        features: list(Scorer)
            scoring features
        coefficients: list(float)
            features' coefficients
        """

        if not len(features) == len(coefficients):
            raise ValueError("there should be as many as many coefficients as there are features.")

        self.features = features
        if type(coefficients) in [list, np.ndarray]:
            self.coefficients = torch.tensor(coefficients)
        else:
            self.coefficients = coefficients

        if torch.Tensor != type(self.coefficients):
            raise TypeError("coefficients should come in a tensor, or a tensorable structure.")

    def log_score(self, samples, context):
        """Log-scores the samples given the context
        using the instance's features and their coefficients

        Parameters
        ----------
        samples : list(str)
            list of samples to log-score
        context: text
            context used for the samples

        Returns
        -------
        tensor of log-scores"""

        device = get_device(self.coefficients)

        feature_log_scores = torch.stack(
                ([feature.log_score(samples, context).to(device) for feature in self.features])
            ) # [n_features, n_samples]
        weighted_log_scores = self.coefficients.repeat(len(samples), 1) * feature_log_scores.t()

        return weighted_log_scores.sum(dim=1)

    def __str__(self):
        return f"ExponentialScorer({self.features}, {self.coefficients})"
