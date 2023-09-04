# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from transformers import pipeline

from .positive_scorer import PositiveScorer


class PipelineScorer(PositiveScorer):
    """
    Feature class relying on the pipelines from Huggingface's transformers
    """

    def __init__(self, label, params, temperature=1.0):
        """initializes a PipelineFeature's instance

        Parameters
        ----------
        label: string
            expected positive label from the pipeline
        """

        self.label = label
        self.pipeline = pipeline(**params)
        self.temperature = temperature

    def score(self, samples, _):
        """computes the scores of the samples
        from the label returned by the pipeline

        Parameters
        ----------
        samples : list(Sample)
            list of samples to log-score

        Returns
        -------
        tensor of scores"""

        return (
                torch.tensor(
                    [[r_i["score"] for r_i in r if self.label == r_i["label"]][0]
                    for r in self.pipeline([s.text for s in samples], return_all_scores=True)]
                ).float()
            ) / self.temperature
