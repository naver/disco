# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os
from torch.utils.tensorboard import SummaryWriter
from .base import BaseTunerObserver
from collections import defaultdict

class TensorBoardLogger(BaseTunerObserver):
    """
    Reports DPGTuner statistics to Neptune
    """

    def __init__(self, tuner, **kwargs):
        """Constructor of a NeptuneLogger object

        Parameters
        ----------
        tuner: DPGTuner
            The tuner object whose statistics we want to report
        """
        super(TensorBoardLogger, self).__init__(tuner)
        self.writer = SummaryWriter(**kwargs)
        self.x_counter = defaultdict(int)

    def __setitem__(self, k, v):
        """
        Report arbitrary parameter/value combinations
        """
        if isinstance(v, str):
            self.writer.add_text(k, v)
        else:
            self.writer.add_scalar(k, v)

    def __exit__(self, *exc):
        self.writer.close()

    def on_parameters_updated(self, params):
        self.writer.add_hparams(params, {})

    def on_metric_updated(self, name, value):
        x = self.x_counter[name]
        self.x_counter[name] += 1
        self.writer.add_scalar(name, value, x)

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        x = self.x_counter['samples']
        self.x_counter['samples'] += 1
        for s in samples[:10]:
            self.writer.add_text('samples', context + s.text, x)

    def on_step_idx_updated(self, s):
        pass

    def on_ministep_idx_updated(self, s):
        pass
        # self.run["ministeps"].log(s)
