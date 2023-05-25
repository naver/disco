# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .base import BaseLoss
import torch
from .misc.ema_baseline import EMABaseline

class JSLoss(BaseLoss):
    """
    Jensen-Shannon divergence loss for DPG
    """
    def __init__(self, use_baseline=False, ema_weight=0.99):
        """
        Parameters
        ----------
        baseline: boolean
            use a baseline to reduce variance
        ema_weight: float
            weight to compute the exponential moving average of the rewards
        """
        super(JSLoss, self).__init__()
        self.use_baseline = use_baseline
        if use_baseline:
            self.baseline = EMABaseline(ema_weight)

    def __call__(self, samples, context, scores, ebm_scores, target_scores, z):
        """
        Computes the JS "loss" on a given minibatch of samples

        Parameters
        ----------
        samples: list of items
            samples from the proposal model
        context: text
            context for the samples
        scores: array of floats
            logprobabilities for the samples according to the proposal
        ebm_scores: array of floats
            logscores for the samples according to the ebm
        target_scores: array of floats
            logprobabilities for the samples according to the target model
        z: float
            estimation of the partition function of the EBM
 
        Returns
        -------
        mean loss across the minibatch
        """

        importance_ratios = torch.exp(target_scores.detach() - scores)
        normalized_ebm_scores = ebm_scores - z.log()
        rewards = importance_ratios * -torch.log((1 + (normalized_ebm_scores - target_scores.detach()).exp()) / 2)
        self.metric_updated.dispatch('importance_ratios', importance_ratios.mean())
        self.metric_updated.dispatch('rewards', rewards.mean())
        if self.use_baseline:
            advantage = self.baseline.advantage(rewards)
            loss = torch.mean(advantage * target_scores)
            self.metric_updated.dispatch('advantage', advantage.mean())
        else:
            loss = torch.mean(rewards * target_scores)

        return loss
