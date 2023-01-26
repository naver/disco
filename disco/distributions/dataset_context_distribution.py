# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import numpy as np
from random import sample
from datasets import load_dataset

from .distribution import Distribution

class DatasetContextDistribution(Distribution):
    """
    Context distribution class, fetching the contexts from a text file.
    It can be used as a template for other context distributions.
    """
    def __init__(self, dataset="", subset="", split="train", key="text", prefix=""):
        """
        Parameters
        ----------
        dataset: string
            name of dataset in Hugging Face's Datasets
        subset: string
            reference of subset in dataset
        split: string
            reference of split in dataset/subset
        key: string
            key to use on row to pick the relevant part
        prefix: text
            text prepended to each context
        """

        try:
            self.dataset = load_dataset(dataset, subset, split=split)
        except IOError:
            self.dataset = list()

        assert self.dataset, "there's an issue with the parameters of the dataset."

        self.key = key
        self.prefix = prefix

    def log_score(self, contexts):
        """Computes plausible log-probabilities of the contexts.
        Note that there's no check that the context are part of the dataset,
        hence the plausible qualifier.

        Parameters
        ----------
        contexts: list(str)
            list of contexts to (log-)score

        Returns
        -------
        tensor of logprobabilities
        """

        assert contexts, "there needs to be contexts to (log-)score."

        return torch.log(torch.full((len(contexts), ), 1 / self.dataset.num_rows))

    def sample(self, sampling_size=32):
        """Samples random elements from the list of contexts
        
        Parameters
        ----------
        sampling_size: int
            number of contexts to sample
        
        Returns
        -------
        tuple of (list of texts, tensor of logprobs)
        """
    
        assert self.dataset.num_rows >= sampling_size, "the dataset does not have enough elements to sample."

        contexts = [self.prefix + c[self.key]\
            for c in self.dataset.select(sample(range(self.dataset.num_rows), sampling_size))]
        return (contexts, self.log_score(contexts))