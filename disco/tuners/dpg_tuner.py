# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .cdpg_tuner import CDPGTuner
from disco.distributions.single_context_distribution import SingleContextDistribution

class DPGTuner(CDPGTuner):
    """
    DPG tuning class,
    a specific case of CDPG with a single, fixed, context.

    The algorithm has been introduced in
    "Distributional Reinforcement Learning for Energy-Based Sequential Models"
    Tetiana Parshakova, Jean-Marc Andreoli, Marc Dymetman
    https://arxiv.org/abs/1912.08517
    """

    def __init__(self, *args, context="", **kwargs):
        """
        Parameters
        ----------
        context: text
            a single textual sequence to contextualize the sampling from the proposal
        """

        super(DPGTuner, self).__init__(
                *args,
                context_distribution=SingleContextDistribution(context),
                context_sampling_size=1,
                **kwargs
            )
