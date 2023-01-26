# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from .base import BaseLoss

class KLLoss(BaseLoss):
    """
    Kullback-Leibler divergence loss for DPG
    """
    def __init__(self, use_baseline=True):
        """
        Parameters
        ----------
        use_baseline: boolean
            use a baseline to reduce variance
        """
        super(KLLoss, self).__init__()
        self.use_baseline = use_baseline

    def __call__(self, samples, context, proposal_log_scores, target_log_scores, model_log_scores, z):
        """
        Computes the KL loss on a given minibatch of samples
        ∇ loss = (target(x) / q(x)) * ∇ log π(x)

        Parameters
        ----------
        samples: list of items
            samples from the proposal network
        context: text
            context for the samples
        proposal_log_scores: array of floats
            log-probabilities for the samples according to the proposal
        target_log_scores: array of floats
            log-probabilities for the samples according to the target
        model_log_scores: array of floats
            log-probabilities for the samples according to the model network
        z: float
            estimation of the partition function of the EBM
 
        Returns
        -------
        mean loss across the minibatch
        """

        normalized_target_log_scores = target_log_scores - torch.log(z)
        rewards = torch.exp(normalized_target_log_scores - proposal_log_scores)
        self.metric_updated.dispatch('rewards', rewards.mean())
        if self.use_baseline:
            importance_ratios = (model_log_scores.detach() - proposal_log_scores).exp()
            advantage = rewards - importance_ratios
            loss = -torch.mean(advantage * model_log_scores)
            self.metric_updated.dispatch('importance_ratios', importance_ratios.mean())
            self.metric_updated.dispatch('advantage', advantage.mean())
        else:
            loss = -torch.mean(rewards * model_log_scores)

        return loss
