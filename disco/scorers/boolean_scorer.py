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
        super().__init__(predicate)

        self.predicate = self._broadcast(lambda s, c: predicate(s, c).float())