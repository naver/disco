# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import mlflow
import tempfile
import json
from pathlib import Path
from .base import BaseTunerObserver
from collections import defaultdict
import torch
import numpy as np

class MLFlowLogger(BaseTunerObserver):
    """
    Reports DPGTuner statistics to MLFlow
    """

    def __init__(self, tuner, project, name=None):
        """Constructor of a WandBLogger object

        Parameters
        ----------
        tuner: DPGTuner
            The tuner object whose statistics we want to report
        project: string
            The MLFlow experiment_id to which we want to report the statistics
        name: string
            The MLFlow run_id to which we want to report the statistics
        """
        super(MLFlowLogger, self).__init__(tuner)
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=name, log_system_metrics=True)
        self.step = None
        self.last_step_eval_samples_reported = None
        self._reset_stats()

    def _reset_stats(self):
        self.stats = defaultdict(list)

    def _report_step_stats(self, step):
        metrics = {}
        for k, vals in self.stats.items():
            if len(vals) == 1:
                metrics[k] = vals[0]
            else:
                try:
                    metrics[f"{k}/min"] = np.min(vals)
                    metrics[f"{k}/max"] = np.max(vals)
                    metrics[f"{k}/mean"] = np.mean(vals)
                except TypeError:
                    raise RuntimeError(f"TypeError while processing {k} = {vals}")
        mlflow.log_metrics(metrics, step=step)

    def __setitem__(self, k, v):
        """
        Report arbitrary parameter/value combinations
        """
        mlflow.log_param(k, v)

    def __exit__(self, *exc):
        self._report_step_stats(self.step)
        self.run.__exit__(*exc)

    def on_metric_updated(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.stats[name].append(value)

    def on_parameters_updated(self, params):
        mlflow.log_params(params)

    def on_eval_samples_updated(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        """
        Logs evaluation samples and their scores as an MLflow artifact (JSON format).
        """
        if self.last_step_eval_samples_reported is not None and self.last_step_eval_samples_reported == self.step:
            # we already reported samples for this step
            return
        self.last_step_eval_samples_reported = self.step
        # Build dict
        data = {
            "context": context,
            "samples": [
                {
                    "text": getattr(s, "text", str(s)),
                    "proposal_log_score": float(proposal_log_scores[i].item()),
                    "model_log_score": float(model_log_scores[i].item()),
                    "target_log_score": float(target_log_scores[i].item())
                }
                for i, s in enumerate(samples)
            ]
        }

        # Write to a temporary JSON file and log
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir, f"{self.step:03}.json")
            json.dump(data, open(tmpfile, 'w'), indent=2)
            mlflow.log_artifact(tmpfile, artifact_path=f"samples")

    def on_step_idx_updated(self, s):
        if self.step is not None:
            self._report_step_stats(self.step)
            self._reset_stats()
        self.step = s
        mlflow.log_metric("steps", s, step=self.step)