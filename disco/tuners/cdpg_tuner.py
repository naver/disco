# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .tuner import Tuner
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.tuners.losses import *


class CDPGTuner(Tuner):
    """Contextual DPG tuning class,
    relying on a ContextDistribution and KLLoss().
    """

    def __init__(self, *args, context_distribution=SingleContextDistribution(), **kwargs):
        """
        Parameters
        ----------
        context_distribution: distribution
            a distribution to contextualize the sampling from the proposal
        """

        super(CDPGTuner, self).__init__(
                *args,
                context_distribution=context_distribution,
                loss=KLLoss(),
                **kwargs
            )