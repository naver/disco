# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from datetime import datetime
from .base import BaseTunerObserver
from tqdm.autonotebook import tqdm


class ConsoleLogger(BaseTunerObserver):
    def __init__(self, tuner):
        super(ConsoleLogger, self).__init__(tuner)
        tuner.proposal_updated.enroll(self.on_proposal_updated)
        self.n = tuner.params["n_gradient_steps"]
        self.step = None

    def __exit__(self, *exc):
        stamp = datetime.now().strftime("%H:%M:%S (%Y/%m/%d)")
        print (f"finished at {stamp}")

    def on_parameters_updated(self, params):
        for k, v in params.items():
            print (f"{k}: ", v)

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        tqdm.write(f"Context: {context}")
        tqdm.write("Samples:")
        n = 3
        tqdm.write("\n".join(s.text + f"\ntarget: {t.item()} - proposal: {p.item()} - model: {m.item()}" for (s, p, m, t) in zip(samples[:n], proposal_log_scores[:n], model_log_scores[:n], target_log_scores[:n])))

    def on_step_idx_updated(self, s):
        self.step = s
        tqdm.write(f"Step {s}/{self.n}")

    def on_proposal_updated(self, proposal, divergence_metric, divergence_target_new, divergence_target_old):
        tqdm.write(f"updating proposal according to {divergence_metric} divergence at step {self.step}: "
                   f"{divergence_target_new} < {divergence_target_old}")