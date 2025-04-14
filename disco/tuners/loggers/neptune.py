# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import neptune
import os
from .base import BaseTunerObserver

def get_proxies():
    proxies = {}
    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    if http_proxy:
        proxies["http"] = http_proxy
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies

class NeptuneLogger(BaseTunerObserver):
    """
    Reports DPGTuner statistics to Neptune
    """

    def __init__(self, tuner, project, name=None, api_token=None, **kwargs):
        """Constructor of a NeptuneLogger object

        Parameters
        ----------
        tuner: DPGTuner
            The tuner object whose statistics we want to report
        project: string
            The Neptune project to which we want to report the statistics
        api_token: string
            The Neptune API token
        """
        super(NeptuneLogger, self).__init__(tuner)
        if not 'proxies' in kwargs:
            kwargs['proxies'] = get_proxies()
        self.run = neptune.init_run(project=project, 
                name=name,
                api_token=api_token,
                **kwargs)

    def __setitem__(self, k, v):
        """
        Report arbitrary parameter/value combinations
        """
        self.run[k] = v

    def __exit__(self, *exc):
        self.run.stop()

    def on_parameters_updated(self, params):
        self.run["parameters"] = params

    def on_metric_updated(self, name, value):
        self.run[name].log(value)

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        self.run["samples"].log([s.text for s in samples[:10]])

    def on_step_idx_updated(self, s):
        self.run["steps"].log(s)