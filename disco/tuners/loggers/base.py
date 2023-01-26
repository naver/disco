# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

class BaseTunerObserver(object):
    def __init__(self, tuner):
        tuner.parameters_updated.enroll(self.on_parameters_updated)
        tuner.metric_updated.enroll(self.on_metric_updated)
        tuner.step_idx_updated.enroll(self.on_step_idx_updated)
        tuner.ministep_idx_updated.enroll(self.on_ministep_idx_updated)
        tuner.eval_samples_updated.enroll(self.on_eval_samples_updated)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def on_parameters_updated(self, params):
        pass

    def on_metric_updated(self, name, value):
        pass

    def on_step_idx_updated(self, s):
        pass

    def on_ministep_idx_updated(self, s):
        pass

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        pass
