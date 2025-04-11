# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from tqdm.autonotebook import trange
from collections import defaultdict
from transformers import (get_constant_schedule_with_warmup,
                        get_linear_schedule_with_warmup,
                        get_cosine_schedule_with_warmup)


from disco.tuners.losses import *
from disco.samplers import AccumulationSampler
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.metrics import KL, TV, JS
from disco.utils.helpers import batchify
from disco.utils.observable import Observable, forward
from disco.utils.device import to_same_device, get_device
from disco.utils.moving_average import MovingAverage
from disco.utils.moving_average import average

divergence_pointwise_estimates_funcs = {
        'tv': TV.pointwise_estimates,
        'kl': KL.pointwise_estimates, 
        'js': JS.pointwise_estimates}


class Tuner():
    """
    Generic tuning class.

    Observables
    -----------
    step_idx: reports the current gradient updates index
        step_idx: integer
    ministep_idx: reports the current minibatch index
        ministep_idx: integer
    metric_updated: reports the value of a given metric
        name: string
        value: scalar
    proposal_updated: reports the new proposal distribution when it is updated
        proposal: Distribution
    eval_samples_updated: reports a fresh set of samples that the network has not yet been trained on
        context: text
        samples: list
        proposal_log_scores: list of floats
        model_log_scores: list of floats
        target_log_scores: list of floats
    """

    default_params = {
        "optimizer": "Adam",
        "learning_rate": 1.41e-5,
        "scheduler": "constant",
        "warmup_steps":2*6,
        "n_gradient_steps": 2**10, # number of gradient updates
        "n_samples_per_step": 2**10, # number of samples used per update step
        "scoring_size": 2**6, # number of samples used for one computation of the loss
        "sampling_size": 2**5, # number of samples requested per sampling
        "context_sampling_size": 2**4, # number of different contexts to sample
        "divergence_evaluation_interval": 2**4, # number of gradient steps between evaluation of divergence
                                                # (also used to eventually update proposal when offline tuning)
        "proposal_update_metric": "kl" # the proposal will be updated if the model is better according to this metric
    }

    def __init__(self, model, target, proposal=None, context_distribution=SingleContextDistribution(), loss=JSLoss(), features=[], 
            track_metrics=["kl", "tv", "js"], track_divergence_from_base=False, **params):
        """
        Parameters
        ----------
        model: distribution
            model distribution, to be tuned
        target: product
            EBM made of a distribution and one or multiple (log-)scorers 
        proposal: distribution  
            sampling distribution, if specified tuning is offline
            else online (model is also used to sample from)
        context_distribution: distribution
            to contextualize the sampling from the proposal
        loss: function
            used to compute of the loss at each step
        features: list of (label, feature)
            feature monitored during the tuning
        track_metrics: list of strings
            metrics used to report differences between the target and the 
            model/proposal distributions.
        track_divergence_from_base: boolean
            whether or not track divergence from the base model of the EBM
        params: dictionary
            fine-tuning parameters
        """
        self.params = self.default_params
        self.params.update(params)
        self.target  = target
        if proposal:
            self.proposal = proposal
            self.learning = "offline"
        else:
            self.proposal = model
            self.learning = "online"
        self.model = model

        self.context_distribution = context_distribution

        self._loss = loss

        self.features = list(features)

        if "AdamW" == self.params["optimizer"]:
            self.optimizer = torch.optim.AdamW(self.model.network.parameters(), lr=self.params["learning_rate"])
        if "SGD" == self.params["optimizer"]:
            self.optimizer = torch.optim.SGD(self.model.network.parameters(), lr=self.params["learning_rate"])
        else:
            self.optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self.params["learning_rate"])

        if "linear" == self.params["scheduler"]:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.params["warmup_steps"])
        elif "cosine" == self.params["scheduler"]:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.params["warmup_steps"], self.params["n_gradient_steps"])
        else:
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.params["warmup_steps"])

        # observables
        self.parameters_updated = Observable()
        self.step_idx_updated = Observable()
        self.ministep_idx_updated = Observable()
        self.metric_updated = Observable()
        self.proposal_updated = Observable()
        self.eval_samples_updated = Observable()

        # setup metrics to track
        forward(self._loss.metric_updated, self.metric_updated)

        if self.params["proposal_update_metric"] not in track_metrics:
            track_metrics.append(self.params["proposal_update_metric"])

        self.z = defaultdict(MovingAverage)
        self.divergence_estimates_target_proposal = dict()
        self.divergence_estimates_target_model = dict()
        for metric in track_metrics:
            assert metric in divergence_pointwise_estimates_funcs, \
                    f"Unknown metric {metric}. " \
                    f"Options are: {list(divergence_pointwise_estimates_funcs.keys())}"
            self.divergence_estimates_target_proposal[metric] = defaultdict(MovingAverage)
            self.divergence_estimates_target_model[metric] = defaultdict(MovingAverage)

        self.track_divergence_from_base = track_divergence_from_base
        if self.track_divergence_from_base:
            self.divergence_estimates_proposal_base = dict()
            self.divergence_estimates_model_base = dict()
            for metric in track_metrics:
                assert metric in divergence_pointwise_estimates_funcs, \
                        f"Unknown metric {metric}. " \
                        f"Options are: {list(divergence_pointwise_estimates_funcs.keys())}"
                self.divergence_estimates_proposal_base[metric] = defaultdict(MovingAverage)
                self.divergence_estimates_model_base[metric] = defaultdict(MovingAverage)

        self.features_moments_proposal = dict()
        self.features_moments_target = dict()
        for (label, feature) in self.features:
            self.features_moments_proposal[label] = defaultdict(MovingAverage)
            self.features_moments_target[label] = defaultdict(MovingAverage)
        if self.features:
            self.eval_samples_updated.enroll(self._update_features_moments)

    def _update_features_moments(self, context, samples, proposal_log_scores, model_log_scores, target_log_scores):
        """
        Improves the importance sampling estimates of the feature moments
        specified on construction of the Tuner

        Parameters
        ----------
        context: text
            context for the samples
        samples: list of items
            samples from the proposal network
        proposal_log_scores: array of floats
            log-probabilities for the samples according to the proposal
        model_log_scores: array of floats
            log-probabilities for the samples according to the model
        target_log_scores: array of floats
            log-probabilities for the samples according to the target
        """
        device = get_device(proposal_log_scores)
        model_log_scores = model_log_scores.to(device)
        logweights = model_log_scores - proposal_log_scores
        importance_ratios = torch.exp(logweights)
        for (label, feature) in self.features:
            proposal_moment_pointwise_estimates = feature.log_score(samples, context).exp().to(device)
            self.features_moments_proposal[label][context] += proposal_moment_pointwise_estimates
            self.features_moments_target[label][context] += importance_ratios * proposal_moment_pointwise_estimates

    def _update_moving_z(self, proposal_log_scores, target_log_scores, context):
        """
        Improves the `z` importance sampling estimate of Z 
        by averaging new samples

        Parameters
        ----------
        proposal_log_scores: array of floats
            log-probabilities of the samples according to the proposal
        target_log_scores: array of floats
            log-probabilities of the samples according to the target
        context: text
            context for the samples
        """
        target_log_scores, proposal_log_scores = to_same_device(target_log_scores, proposal_log_scores)

        z_pointwise_estimates = torch.exp(target_log_scores - proposal_log_scores)
        self.z[context] += z_pointwise_estimates
        self.metric_updated.dispatch('z', average(self.z))

    def _update_divergence_estimates_target_proposal(self, proposal_log_scores, target_log_scores, context):
        """
        Improves the importance sampling estimate of D(p||q)
        for every divergence D by averaging new samples

        Parameters
        ----------
        proposal_log_scores: array of floats
            log-probabilities of the samples according to the proposal
        target_log_scores: array of floats
            log-probabilities of the samples according to the target
        context: text
            context for the samples
        """
        target_log_scores, proposal_log_scores = to_same_device(target_log_scores, proposal_log_scores)

        if self.z[context].value > 0:
            for divergence_type, _ in self.divergence_estimates_target_proposal.items():
                self.divergence_estimates_target_proposal[divergence_type][context] += \
                    divergence_pointwise_estimates_funcs[divergence_type](
                        target_log_scores, proposal_log_scores, self.z[context].value)

    def _update_divergence_estimates_target_model(self, proposal_log_scores, target_log_scores, model_log_scores, context):
        """
        Improves the importance sampling estimates of D(p||q)
        for every divergence D by averaging new samples

        Parameters
        ----------
        proposal_log_scores: array of floats
            log-probabilities of the samples according to the proposal
        target_log_scores: array of floats
            log-probabilities of the samples according to the target
        model_log_scores: array of floats
            log-probabilities of the samples according to the model
        context: text
            context for the samples
        """
        target_log_scores, model_log_scores, proposal_log_scores = to_same_device(
                target_log_scores, model_log_scores, proposal_log_scores)

        if self.z[context].value > 0:
            for divergence_type, _ in self.divergence_estimates_target_model.items():
                self.divergence_estimates_target_model[divergence_type][context] += \
                    divergence_pointwise_estimates_funcs[divergence_type](
                        target_log_scores, model_log_scores, self.z[context].value,
                        proposal_log_scores=proposal_log_scores)

    def _update_divergence_estimates_proposal_base(self, proposal_log_scores, base_log_scores, context):
        """
        Improves the importance sampling estimate of D(p||q)
        for every divergence D by averaging new samples

        Parameters
        ----------
        proposal_log_scores: array of floats
            log-probabilities of the samples according to the proposal
        base_log_scores: array of floats
            log-probabilities of the samples according to the base
        context: text
            context for the samples
        """
        base_log_scores, proposal_log_scores = to_same_device(base_log_scores, proposal_log_scores)

        for divergence_type, _ in self.divergence_estimates_target_proposal.items():
            self.divergence_estimates_proposal_base[divergence_type][context] += \
                divergence_pointwise_estimates_funcs[divergence_type](
                    proposal_log_scores, base_log_scores, torch.as_tensor(1), proposal_log_scores)

    def _update_divergence_estimates_model_base(self, proposal_log_scores, model_log_scores, base_log_scores, context):
        """
        Improves the importance sampling estimate of D(p||q)
        for every divergence D by averaging new samples

        Parameters
        ----------
        proposal_log_scores: array of floats
            log-probabilities of the samples according to the proposal
        model_log_scores: array of floats
            log-probabilities of the samples according to the model
        base_log_scores: array of floats
            log-probabilities of the samples according to the base
        context: text
            context for the samples
        """
        model_log_scores, base_log_scores, proposal_log_scores = \
                to_same_device(model_log_scores, base_log_scores, proposal_log_scores)

        for divergence_type, _ in self.divergence_estimates_target_proposal.items():
            self.divergence_estimates_model_base[divergence_type][context] += \
                divergence_pointwise_estimates_funcs[divergence_type](
                    model_log_scores, base_log_scores, torch.as_tensor(1), proposal_log_scores)

    def _report_and_reset_importance_sampling_estimate(self, estimates_dict, distributions_name):
        """
        Reports all tracked metrics in the estimates_dict using
        as key name a concatentation of the metric name and the distributions_name

        estimates_dict: dictionary (string, dictionary(string, MovingAverage))
            The dictionary tracking metric estimates for each context
        distributions_name: string
            A name that identifies the distributions of which we are tracking the metric
        """
        for metric_name, moving_averages in estimates_dict.items():
            self.metric_updated.dispatch(f"{metric_name}_{distributions_name}",
                    average(moving_averages))
            estimates_dict[metric_name] = defaultdict(MovingAverage)

    def _update_proposal_if_better(self):
        """
            Checks if D(p||.) is lower for model than for the proposal
            and if so, updates the proposal
        """
        if average(self.divergence_estimates_target_proposal[self.params["proposal_update_metric"]]) > \
                average(self.divergence_estimates_target_model[self.params["proposal_update_metric"]]):
            print("updating proposal according to KL divergence")
            self.proposal.network.load_state_dict(self.model.network.state_dict())
            self.metric_updated.dispatch('proposal_updated', 1)
            self.proposal_updated.dispatch(self.proposal)
        else:
            self.metric_updated.dispatch('proposal_updated', 0)

    def _compute_gradient(self, samples, proposal_log_scores, target_log_scores, model_log_scores, context, n_steps):
        """
        Computes the gradient on a minibatch of samples

        Parameters
        ----------
        samples: list of items
            samples from the proposal network
        proposal_log_scores: array of floats
            log-probabilities for the samples according to the proposal
        target_log_scores: array of floats
            log-probabilities for the samples according to the target
        model_log_scores: array of floats
            log-probabilities for the samples according to the model
        context: text
            context for the samples
        n_steps: int
            number of accumulation steps
        """
        proposal_log_scores, target_log_scores, model_log_scores, z_value = to_same_device(
                proposal_log_scores, target_log_scores, model_log_scores, self.z[context].value)

        if z_value > 0:
            loss = self._loss(samples, context, proposal_log_scores, target_log_scores, model_log_scores, z_value) / n_steps
            self.metric_updated.dispatch('loss', loss.item())
            loss.backward()

    def _step(self):
        """
        Performs a tuning step of the model distribution's network

        Performs a single step of gradient updates on a batch of samples:
          - obtains samples and their log-scores from the proposal network
          - repeats gradient computations, with minibatches
          - applies the accumulated gradients
    
        """
        sampler = AccumulationSampler(self.proposal, total_size=self.params["n_samples_per_step"])
        n_steps = self.params["n_samples_per_step"] // self.params["scoring_size"]

        contexts, _ = self.context_distribution.sample(self.params["context_sampling_size"])

        for context in contexts:

            samples, proposal_log_scores = sampler.sample(sampling_size=self.params["sampling_size"], context=context)
            target_log_scores = batchify(self.target.log_score, self.params["scoring_size"], samples=samples, context=context)

            self._update_moving_z(proposal_log_scores, target_log_scores, context)
            self._update_divergence_estimates_target_proposal(proposal_log_scores, target_log_scores, context)
            if self.track_divergence_from_base:
                base = self.target.scorers[0]
                base_log_scores = batchify(base.log_score, self.params["scoring_size" ], samples=samples, context=context)
                self._update_divergence_estimates_proposal_base(proposal_log_scores, base_log_scores, context)

            for s in range(n_steps):
                self.ministep_idx_updated.dispatch(s)
                minibatch_slice = slice(s * self.params["scoring_size"], (s + 1) * self.params["scoring_size"])

                mb_samples = samples[minibatch_slice]
                mb_proposal_log_scores = proposal_log_scores[minibatch_slice]
                mb_target_log_scores = target_log_scores[minibatch_slice]
                mb_model_log_scores = self.model.log_score(mb_samples, context=context, grad=True)

                self._update_divergence_estimates_target_model(
                        mb_proposal_log_scores, mb_target_log_scores, mb_model_log_scores, context)
                if self.track_divergence_from_base:
                    mb_base_log_scores = base_log_scores[minibatch_slice]
                    self._update_divergence_estimates_model_base(mb_proposal_log_scores, mb_model_log_scores, mb_base_log_scores, context)
                self.eval_samples_updated.dispatch(
                        context, mb_samples, mb_proposal_log_scores, mb_model_log_scores, mb_target_log_scores)

                self._compute_gradient(
                    mb_samples, mb_proposal_log_scores, mb_target_log_scores, mb_model_log_scores,
                    context, n_steps * self.params["context_sampling_size"])
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def tune(self):
        """
        Fine-tunes model distribution's network

        Fine-tunes the network of the model distribution:
          - repeats n_gradient_steps tuning steps
          - eventually updates the samplee according to KL divergence
        """
        self.parameters_updated.dispatch(self.params)
        torch.cuda.empty_cache()

        with trange(self.params["n_gradient_steps"], desc='fine-tuning', position=0) as t:
            for s in t:
                self.step_idx_updated.dispatch(s)
                self._step()

                if  0 == (s + 1) % self.params["divergence_evaluation_interval"]:
                    if "offline" == self.learning:
                        self._update_proposal_if_better()
                    self._report_and_reset_importance_sampling_estimate(self.divergence_estimates_target_model, 'target_model')
                    self._report_and_reset_importance_sampling_estimate(self.divergence_estimates_target_proposal, 'target_proposal')
                    if self.track_divergence_from_base:
                        self._report_and_reset_importance_sampling_estimate(self.divergence_estimates_model_base, 'model_base')
                        self._report_and_reset_importance_sampling_estimate(self.divergence_estimates_proposal_base, 'proposal_base')
                    self._report_and_reset_importance_sampling_estimate(self.features_moments_proposal, 'proposal')
                    self._report_and_reset_importance_sampling_estimate(self.features_moments_target, 'target')
