# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .tuner import Tuner
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.tuners.losses import *


class FDPGTuner(Tuner):
    """Unconditional f-DPG tuning class. The algorithm was introduced in

    "Aligning Language Models with Preferences through f-divergence Minimization."
    Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, Marc Dymetman.
    https://arxiv.org/abs/2302.08215
    """

    def __init__(self, *args, context='',
            loss=JSLoss(), **kwargs):
        """
        Parameters
        ----------
        context: text
            a single textual sequence to contextualize the sampling from the proposal
        loss: functor object
            used to compute of the loss at each step
        """

        super(FDPGTuner, self).__init__(
                *args,
                context_distribution=SingleContextDistribution(context),
                n_contexts_per_step=1,
                loss=loss,
                **kwargs
            )
