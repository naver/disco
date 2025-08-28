# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
from tqdm.auto import trange, tqdm
from collections import defaultdict
from transformers import (get_constant_schedule_with_warmup,
                        get_linear_schedule_with_warmup,
                        get_cosine_schedule_with_warmup)


from disco.tuners.losses import *
from disco.samplers import AccumulationSampler
from disco.metrics import KL, TV, JS
from disco.utils.helpers import batchify
from disco.utils.observable import Observable, forward
from disco.utils.device import to_same_device, get_device
from disco.utils.moving_average import WindowedMovingAverage, MovingAverage
from disco.utils.moving_average import average
from disco.utils.timer import Timer

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
        "n_samples_per_context": 2**10, # number of samples generated for each context
        "scoring_size": 2**6, # number of samples used for one computation of the loss
        "sampling_size": 2**5, # number of samples requested per sampling
        "n_contexts_per_step": 2**4, # number of different contexts to sample per gradient step
        "proposal_update_interval": 2**4, # number of gradient steps every which to update the proposal in offline learning
        "proposal_update_metric": "kl", # the proposal will be updated if the model is better according to this metric
        "estimates_rolling_window_size": 2**8, # number of samples to accumulate in a rollowing window estimate
        "validation_interval": 2**4, # number of steps every which to compute feature moments on the validation contexts
        "validate_on_start": True,
    }

    def __init__(self, model, target, proposal=None, context_dataset=None, val_dataset=None, loss=JSLoss(), features=[],
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
        torch.autograd.set_detect_anomaly(True)
        self.params = self.default_params
        self.params.update(params)
        self.params['scoring_size'] = min(self.params['scoring_size'], self.params['n_samples_per_context'])
        self.params['sampling_size'] = min(self.params['sampling_size'], self.params['n_samples_per_context'])
        self.target  = target
        if proposal:
            self.proposal = proposal
            self.learning = "offline"
        else:
            self.proposal = model
            self.learning = "online"
        self.model = model

        self.context_dataloader = torch.utils.data.DataLoader(context_dataset, batch_size=self.params["n_contexts_per_step"], shuffle=False, collate_fn=lambda x: x)
        self.context_iterator = iter(self.context_dataloader)

        self.val_dataset = val_dataset

        self._loss = loss

        self.features = list(features)

        if "AdamW" == self.params["optimizer"]:
            self.optimizer = torch.optim.AdamW(self.model.network.parameters(), lr=self.params["learning_rate"])
        if "SGD" == self.params["optimizer"]:
            self.optimizer = torch.optim.SGD(self.model.network.parameters(), lr=self.params["learning_rate"])
        else:
            self.optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self.params["learning_rate"])

        if "linear" == self.params["scheduler"]:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.params["warmup_steps"], self.params["n_gradient_steps"])
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
            self.divergence_estimates_target_proposal[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
            self.divergence_estimates_target_model[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])

        self.track_divergence_from_base = track_divergence_from_base
        if self.track_divergence_from_base:
            self.divergence_estimates_proposal_base = dict()
            self.divergence_estimates_model_base = dict()
            for metric in track_metrics:
                assert metric in divergence_pointwise_estimates_funcs, \
                        f"Unknown metric {metric}. " \
                        f"Options are: {list(divergence_pointwise_estimates_funcs.keys())}"
                self.divergence_estimates_proposal_base[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
                self.divergence_estimates_model_base[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])

        self.features_moments_proposal = dict()
        self.features_moments_model = dict()
        for (label, feature) in self.features:
            self.features_moments_proposal[label] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
            self.features_moments_model[label] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
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
            proposal_moment_pointwise_estimates = feature.log_score(samples, context=context).exp().to(device)
            self.features_moments_proposal[label].update(proposal_moment_pointwise_estimates)
            self.features_moments_model[label].update(importance_ratios * proposal_moment_pointwise_estimates)

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
        self.z[context].update(z_pointwise_estimates)
        self.metric_updated.dispatch('z', self.z[context].value)

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
                self.divergence_estimates_target_proposal[divergence_type].update(
                    divergence_pointwise_estimates_funcs[divergence_type](
                        target_log_scores, proposal_log_scores, self.z[context].value))

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
                self.divergence_estimates_target_model[divergence_type].update(
                    divergence_pointwise_estimates_funcs[divergence_type](
                        target_log_scores, model_log_scores, self.z[context].value,
                        proposal_log_scores=proposal_log_scores))

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
            self.divergence_estimates_proposal_base[divergence_type].update(
                divergence_pointwise_estimates_funcs[divergence_type](
                    proposal_log_scores, base_log_scores, torch.as_tensor(1), proposal_log_scores))

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
            self.divergence_estimates_model_base[divergence_type].update(
                divergence_pointwise_estimates_funcs[divergence_type](
                    model_log_scores, base_log_scores, torch.as_tensor(1), proposal_log_scores))

    def _report_importance_sampling_estimates(self, estimates_dict, distributions_name):
        """
        Reports all tracked metrics in the estimates_dict using
        as key name a concatentation of the metric name and the distributions_name

        estimates_dict: dictionary (string, dictionary(string, MovingAverage))
            The dictionary tracking metric estimates for each context
        distributions_name: string
            A name that identifies the distributions of which we are tracking the metric
        context: string
            The specific context for which to report the estimate
        """
        for metric_name, metric_estimates in estimates_dict.items():
            if metric_estimates.value is not None:
                self.metric_updated.dispatch(f"{metric_name}_{distributions_name}",
                        metric_estimates.value)

    def _report_all_importance_sampling_estimates(self, context):
        """
        Reports all tracked metrics for a given context
        """
        self._report_importance_sampling_estimates(self.divergence_estimates_target_model, 'target_model')
        self._report_importance_sampling_estimates(self.divergence_estimates_target_proposal, 'target_proposal')
        if self.track_divergence_from_base:
            self._report_importance_sampling_estimates(self.divergence_estimates_model_base, 'model_base')
            self._report_importance_sampling_estimates(self.divergence_estimates_proposal_base, 'proposal_base')
        self._report_importance_sampling_estimates(self.features_moments_proposal, 'proposal')
        self._report_importance_sampling_estimates(self.features_moments_model, 'model')

    def _update_proposal_if_better(self):
        """
            Checks if D(p||.) is lower for model than for the proposal
            and if so, updates the proposal
        """
        divergence_target_proposal = average(self.divergence_estimates_target_proposal[self.params["proposal_update_metric"]])
        divergence_target_model = average(self.divergence_estimates_target_model[self.params["proposal_update_metric"]])
        if divergence_target_proposal > \
                divergence_target_model:
            self.proposal.network.load_state_dict(self.model.network.state_dict())
            self.metric_updated.dispatch('proposal_updated', 1)
            self.proposal_updated.dispatch(self.proposal, self.params['proposal_update_metric'],
                                            divergence_target_model, divergence_target_proposal)
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
        proposal_log_scores, target_log_scores, model_log_scores = to_same_device(
                proposal_log_scores, target_log_scores, model_log_scores)
        z_value = self.z[context].value

        if z_value > 0:
            z = torch.tensor(z_value, device=model_log_scores.device, dtype=model_log_scores.dtype)
            loss = self._loss(samples, context, proposal_log_scores, target_log_scores, model_log_scores, z) / n_steps
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
        # counters for statistics
        n_samples_to_boost, n_samples_to_downweight = 0, 0

        # iterate over n_contexts_per_step contexts
        contexts = self._get_next_context_batch()
        for context in tqdm(contexts, desc="contexts", leave=False):

            # obtain samples conditioned on this context
            sampler = AccumulationSampler(self.proposal, total_size=self.params["n_samples_per_context"])
            with Timer() as t_sampling:
                samples, proposal_log_scores = sampler.sample(sampling_size=self.params["sampling_size"], context=context)
            self.metric_updated.dispatch("timing/generation_per_sample", t_sampling.elapsed / len(samples))

            # score the samples according to the target distribution
            with Timer() as t_scoring:
                target_log_scores = batchify(self.target.log_score, self.params["scoring_size"], samples=samples, context=context)
            self.metric_updated.dispatch("timing/target_scoring_per_sample", t_scoring.elapsed / len(samples))

            # update importance sampling statistics
            self._update_moving_z(proposal_log_scores, target_log_scores, context)
            self._update_divergence_estimates_target_proposal(proposal_log_scores, target_log_scores, context)
            if self.track_divergence_from_base:
                base = self.target.scorers[0]
                base_log_scores = batchify(base.log_score, self.params["scoring_size" ], samples=samples, context=context)
                self._update_divergence_estimates_proposal_base(proposal_log_scores, base_log_scores, context)

            # peform n_steps forward/backward passes to accumulate gradients
            n_steps = self.params["n_samples_per_context"] // self.params["scoring_size"]
            for s in trange(n_steps, desc="mini-steps", leave=False):
                self.ministep_idx_updated.dispatch(s)
                minibatch_slice = slice(s * self.params["scoring_size"], (s + 1) * self.params["scoring_size"])

                mb_samples = samples[minibatch_slice]
                mb_proposal_log_scores = proposal_log_scores[minibatch_slice]
                mb_target_log_scores = target_log_scores[minibatch_slice]
                mb_model_log_scores = self.model.log_score(mb_samples, context=context, grad=True)

                # also report policy dependent scores
                self._update_divergence_estimates_target_model(
                        mb_proposal_log_scores, mb_target_log_scores, mb_model_log_scores.detach(), context)
                if self.track_divergence_from_base:
                    mb_base_log_scores = base_log_scores[minibatch_slice]
                    self._update_divergence_estimates_model_base(mb_proposal_log_scores, mb_model_log_scores.detach(), mb_base_log_scores, context)
                self.eval_samples_updated.dispatch(
                        context, mb_samples, mb_proposal_log_scores, mb_model_log_scores.detach(), mb_target_log_scores)

                with Timer() as t_backprop:
                    self._compute_gradient(
                        mb_samples, mb_proposal_log_scores, mb_target_log_scores, mb_model_log_scores,
                        context, n_steps * self.params["n_contexts_per_step"])
                self.metric_updated.dispatch("timing/backpropagation_per_sample", t_backprop.elapsed / len(mb_samples))

                mb_in_support = ~mb_target_log_scores.isneginf()
                mb_in_support_target_norm_log_scores = mb_target_log_scores[mb_in_support] - \
                    torch.log(torch.tensor(self.z[context].value, device=mb_target_log_scores.device))
                n_samples_to_boost += (mb_in_support_target_norm_log_scores > mb_model_log_scores[mb_in_support]).sum().item()
                n_samples_to_downweight += (mb_in_support_target_norm_log_scores < mb_model_log_scores[mb_in_support]).sum().item()

            self._report_all_importance_sampling_estimates(context)

        self.metric_updated.dispatch("n_samples_to_boost", n_samples_to_boost)
        self.metric_updated.dispatch("n_samples_to_downweight", n_samples_to_downweight)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.metric_updated.dispatch('lr', self.scheduler.get_last_lr()[0])

    def _get_next_context_batch(self):
        try:
            batch = next(self.context_iterator)
        except StopIteration:
            # Restart the iterator when dataset is exhausted
            self.context_iterator = iter(self.context_dataloader)
            batch = next(self.context_iterator)
        return batch

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

                if self.val_dataset and 0 == s % self.params["validation_interval"] and \
                    (s > 0 or self.params["validate_on_start"]):
                    with Timer() as val_t:
                        self._report_features_on_validation_contexts()
                    self.metric_updated.dispatch("timing/validation", val_t.elapsed)

                with Timer() as step_t:
                    self._step()
                self.metric_updated.dispatch("timing/step", step_t.elapsed)

                if  0 == (s + 1) % self.params["proposal_update_interval"]:
                    if "offline" == self.learning:
                        self._update_proposal_if_better()

    def _report_features_on_validation_contexts(self):
        """
            Compute features moments on contexts extracted from the validation set
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params["sampling_size"], shuffle=False, collate_fn=lambda x: x)
        val_model = self.model.validation()
        features_moments = dict()
        for (label, feature) in self.features:
            features_moments[label] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
        for val_contexts in tqdm(val_dataloader, desc="validation", leave=False):
            samples, _ = val_model.sample_batch(val_contexts, sampling_size=1)
            for (label, feature) in self.features:
                for sample, context in zip(samples, val_contexts):
                    proposal_moment_pointwise_estimate = feature.log_score([sample], context=context).exp()
                    features_moments[label].update(proposal_moment_pointwise_estimate)
        for (label, _) in self.features:
            self.metric_updated.dispatch(f"val_{label}", features_moments[label].value)