# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import wandb
import os
from .base import BaseTunerObserver

class WandBLogger(BaseTunerObserver):
    """
    Reports DPGTuner statistics to Weights & Biases
    """

    def __init__(self, tuner, project, name=None):
        """Constructor of a WandBLogger object

        Parameters
        ----------
        tuner: DPGTuner
            The tuner object whose statistics we want to report
        project: string
            The W&B project to which we want to report the statistics
        """
        super(WandBLogger, self).__init__(tuner)
        self.run = wandb.init(project=project, 
                name=name)

    def __setitem__(self, k, v):
        """
        Report arbitrary parameter/value combinations
        """
        wandb.log({k: v})

    def __exit__(self, *exc):
        self.run.finish()

    def on_parameters_updated(self, params):
        wandb.config.update(params)

    def on_metric_updated(self, name, value):
        wandb.log({name: value})

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        wandb.log({"samples": [s.text for s in samples[:10]]})

    def on_step_idx_updated(self, s):
        wandb.log({"steps": s})