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
    def __init__(self, scoring_function):
        super().__init__(scoring_function)

    def __mul__(self, ot):
        """enables the use of the multiplication sign (*)
        to compose positive scorers"""

        return Product(self, ot)

    def log_score(self, samples, context):
        """returns the log-scores for the samples
        given the context by taking the log of their scores

        Parameters
        ----------
        samples : list(Sample)
            list of samples to log-score
        context: text
            context that the samples relate to

        Returns
        -------
        tensor of log-scores for the samples"""

        return torch.log(self.score(samples, context=context))

    def log_score_batch(self, samples, contexts):
        """
        Returns the batched log-scores for samples against multiple contexts.

        Parameters
        ----------
        samples : list
            A flat list of sample objects.
        contexts : list of str
            A list of contextual text strings.
        n_repeats : int
            The number of samples associated with each context.

        Returns
        -------
        torch.Tensor
            A tensor of log-scores with shape `(num_contexts, n_repeats)`.
        """
        scores = self.score_batch(samples, contexts=contexts)
        return torch.log(scores)

    def score(self, samples, context):
        """relies on the instance's scoring function
        to compute the scores of the samples given the context

        Parameters
        ----------
        samples : list(Sample)
            list of samples to score
        context: text
            context that the samples relate to

        Returns
        -------
        tensor of scores for the samples"""

        return self.scoring_function(samples, context)

    def score_batch(self, samples, contexts):
        """
        Computes scores for a batch of samples against multiple contexts.

        This default implementation works by iterating through the contexts,
        slicing the corresponding samples, and calling the single-context
        `scoring_function` for each.

        Parameters
        ----------
        samples : list
            A flat list of sample objects.
        contexts : list of str
            A list of contextual text strings.

        Returns
        -------
        torch.Tensor
            A tensor of scores with shape `(num_contexts, n_samples_per_context)`.
        """
        n_samples_per_context = len(samples) // len(contexts)
        all_scores = []
        for i, context in enumerate(contexts):
            # Slice the flat list of samples to get the ones for the current context
            context_samples = samples[i]

            if not context_samples:
                continue

            # Call the existing single-context scoring function
            context_scores = self.scoring_function(context_samples, context)
            all_scores.append(context_scores)

        if not all_scores:
            return torch.empty(len(contexts), 0)

        # Stack the list of 1D tensors into a single 2D tensor
        return torch.stack(all_scores, dim=0)


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

        log_scores = [s.log_score(samples, context=context).to(device) for s in self.scorers]
        return reduce(lambda x,y: x+y, log_scores)

    def log_score_batch(self, samples, contexts):
        """
        Computes the product of scores in a batched manner by adding the
        batched log-scores from individual scorers.

        Parameters
        ----------
        samples : list
            A flat list of sample objects.
        contexts : list of str
            A list of contextual text strings.

        Returns
        -------
        torch.Tensor
            A tensor of the summed log-scores with shape `(num_contexts, n_samples_per_context)`.
        """
        try:
            device = self.scorers[0].device
        except AttributeError:
            device = "cpu"

        # Call the batched method on each scorer
        log_scores = [
            s.log_score_batch(samples, contexts=contexts).to(device)
            for s in self.scorers
        ]

        # Sum the resulting tensors element-wise
        return reduce(lambda x, y: x + y, log_scores)

    def __str__(self):
        scorers_str = ", ".join((str(scorer) for scorer in self.scorers))
        return f"Product({scorers_str})"