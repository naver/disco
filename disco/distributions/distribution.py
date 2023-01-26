# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from abc import abstractmethod
import torch

from disco.scorers.positive_scorer import PositiveScorer


class Distribution(PositiveScorer):
    """
    Abstract distribution class, a core entity which can
    be introduced as a PositiveScorer that can produce samples.
    """

    @abstractmethod
    def sample(self, context):
        """Produces samples for the context from the distribution.
        """
        pass