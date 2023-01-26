# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os
from .base import BaseTunerObserver
import json
from pathlib import Path
import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class JSONLogger(BaseTunerObserver):
    """
    Reports DPGTuner statistics to a JSON file
    """

    def __init__(self, tuner, project, name, path=os.environ['DISCO_SAVE_PATH'],
            save_steps=1, store_eval_samples=False, **kwargs):
        """Constructor of a JSONLogger object

        Parameters
        ----------
        tuner: DPGTuner
            The tuner object whose statistics we want to report
        path: string/Path
            The path where we want to store the logs
        project: string
            The subfolder where to store the logs
        name: string
            The filename to which we want to report the statistics
        save_steps: integer
            Number of gradient steps every which to write the json data to disk
        store_eval_samples: boolean
            Whether or not to store the samples in the json file
        """
        super(JSONLogger, self).__init__(tuner)
        self.filename = Path(path) / project / f"{name}.json"
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(list)
        self.save_steps = save_steps
        self.store_eval_samples = store_eval_samples

    def __exit__(self, *exc):
        self.save()

    def save(self):
        with open(self.filename, 'w') as fout:
            json.dump(self.data, fout)

    def __setitem__(self, k, v):
        """
        Report arbitrary parameter/value combinations
        """
        if isinstance(v, Path):
            v = str(v)  # avoid serialization error
        elif isinstance(v, torch.Tensor):
            v = v.item()
        self.data[k] = v

    def on_parameters_updated(self, params):
        self.data["parameters"] = dict(params)

    def on_metric_updated(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(value)

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        if not self.store_eval_samples:
            return
        self.data["samples"].append([s.text for s in samples])
        self.data["samples_ids"].append([s.token_ids.tolist() for s in samples])
        self.data["proposal_scores"].append(proposal_log_scores.tolist())
        self.data["target_scores"].append(target_log_scores.tolist())
        self.data["model_scores"].append(model_log_scores.tolist())

    def on_step_idx_updated(self, s):
        self.data["steps"] = s
        if self.save_steps > 0 and (s % self.save_steps) == 0:
            self.save()

    def on_ministep_idx_updated(self, s):
        self.data["ministeps"] = s
