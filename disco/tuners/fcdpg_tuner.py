# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .tuner import Tuner
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.tuners.losses import *


class FCDPGTuner(Tuner):
    """Contextual f-DPG tuning class. The algorithm was introduced in

    "Aligning Language Models with Preferences through f-divergence Minimization."
    Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, Marc Dymetman.
    https://arxiv.org/abs/2302.08215
    """

    def __init__(self, *args, context_distribution=SingleContextDistribution(),
            loss=JSLoss(), **kwargs):
        """
        Parameters
        ----------
        context_distribution: distribution
            a distribution to contextualize the sampling from the proposal
        loss: functor object
            used to compute of the loss at each step
        """

        super(FCDPGTuner, self).__init__(
                *args,
                context_distribution=context_distribution,
                loss=loss,
                **kwargs
            )
