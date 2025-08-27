# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from .base import BaseLoss
from disco.utils.moving_average import WindowedMovingAverage

class FDivergenceLoss(BaseLoss):
    """
    Kullback-Leibler divergence loss for DPG
    """
    def __init__(self, use_baseline=True, baseline_window_size=1024):
        """
        Parameters
        ----------
        use_baseline: boolean
            use a baseline to reduce variance
        """
        super(FDivergenceLoss, self).__init__()
        if use_baseline:
            self.baseline = WindowedMovingAverage(baseline_window_size)
        else:
            self.baseline = None

    def __call__(self, samples, context, proposal_log_scores, target_log_scores, policy_log_scores, z):
        """
        Computes the KL loss on a given minibatch of samples
        ∇ loss = π(x) / q(x) * f'(π(x) / p(x))) * ∇ log π(x)

        Parameters
        ----------
        samples: list of items
            samples from the proposal network
        context: text
            context for the samples
        proposal_log_scores: 1-dim Tensor
            log-probabilities for the samples according to the proposal
        target_log_scores: 1-dim Tensor
            log-probabilities for the samples according to the target unnormalized distribution (EBM)
        policy_log_scores: 1-dim Tensor
            log-probabilities for the samples according to the policy network
        z: 0-dim Tensor
            estimation of the partition function of the target distribution

        Returns
        -------
        mean loss across the minibatch
        """
        norm_target_log_scores = target_log_scores - torch.log(z)

        log_t = policy_log_scores.detach() - norm_target_log_scores

        # implemented in derived class depending on the desired f
        f_prime = self.f_prime(log_t)

        pseudo_reward = -f_prime

        for r in pseudo_reward:
            self.metric_updated.dispatch('pseudo_reward', r.item())

        if self.baseline is not None:
            if self.baseline.value is not None:
                advantage = pseudo_reward - self.baseline.value
            else:
                advantage = pseudo_reward

            self.baseline.update(pseudo_reward)

            self.metric_updated.dispatch("baseline", self.baseline.value)
        else:
            advantage = pseudo_reward


        for a in advantage:
            self.metric_updated.dispatch('advantage', a.item())

        importance_ratios = (policy_log_scores.detach() - proposal_log_scores).exp()

        for ir in importance_ratios:
            self.metric_updated.dispatch('importance_ratios', ir.item())

        loss = (importance_ratios * (-advantage) * policy_log_scores).mean()

        return loss