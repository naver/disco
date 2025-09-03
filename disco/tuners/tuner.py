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
from disco.utils.helpers import score_in_chunks_batched
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
    eval_samples_updated: reports a fresh set of samples that the model has not yet been trained on
        context: text
        samples: list
        proposal_log_scores: list of floats
        policy_log_scores: list of floats
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
        "proposal_update_metric": "kl", # the proposal will be updated if the policy is better according to this metric
        "estimates_rolling_window_size": 2**8, # number of samples to accumulate in a rollowing window estimate
        "validation_interval": 2**4, # number of steps every which to compute feature moments on the validation contexts
        "validate_on_start": True,
    }

    def __init__(self, policy, target, proposal=None, context_dataset=None, val_dataset=None, loss=JSLoss(), features=[],
            track_metrics=["kl", "tv", "js"], track_divergence_from_base=False, **params):
        """
        Parameters
        ----------
        policy: distribution
            policy distribution, to be tuned
        target: product
            EBM made of a distribution and one or multiple (log-)scorers
        proposal: distribution
            sampling distribution, if specified tuning is offline
            else online (policy is also used to sample from)
        context_distribution: distribution
            to contextualize the sampling from the proposal
        loss: function
            used to compute of the loss at each step
        features: list of (label, feature)
            feature monitored during the tuning
        track_metrics: list of strings
            metrics used to report differences between the target and the
            policy/proposal distributions.
        track_divergence_from_base: boolean
            whether or not track divergence from the base model of the EBM
        params: dictionary
            fine-tuning parameters
        """
        torch.autograd.set_detect_anomaly(True)
        self.params = self.default_params
        self.params.update(params)
        self.params['scoring_size'] = min(self.params['scoring_size'], self.params['n_contexts_per_step'] * self.params['n_samples_per_context'])
        self.params['sampling_size'] = min(self.params['sampling_size'], self.params['n_contexts_per_step'] * self.params['n_samples_per_context'])
        self.target  = target
        if proposal:
            self.proposal = proposal
            self.learning = "offline"
        else:
            self.proposal = policy
            self.learning = "online"
        self.policy = policy

        self.context_dataloader = torch.utils.data.DataLoader(context_dataset, batch_size=self.params["n_contexts_per_step"], shuffle=False, collate_fn=lambda x: x)
        self.context_iterator = iter(self.context_dataloader)

        self.val_dataset = val_dataset

        self._loss = loss

        self.features = list(features)

        if "AdamW" == self.params["optimizer"]:
            self.optimizer = torch.optim.AdamW(self.policy.model.parameters(), lr=self.params["learning_rate"])
        if "SGD" == self.params["optimizer"]:
            self.optimizer = torch.optim.SGD(self.policy.model.parameters(), lr=self.params["learning_rate"])
        else:
            self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=self.params["learning_rate"])

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
        self.divergence_estimates_target_policy = dict()
        for metric in track_metrics:
            assert metric in divergence_pointwise_estimates_funcs, \
                    f"Unknown metric {metric}. " \
                    f"Options are: {list(divergence_pointwise_estimates_funcs.keys())}"
            self.divergence_estimates_target_proposal[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
            self.divergence_estimates_target_policy[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])

        self.track_divergence_from_base = track_divergence_from_base
        if self.track_divergence_from_base:
            self.divergence_estimates_proposal_base = dict()
            self.divergence_estimates_policy_base = dict()
            for metric in track_metrics:
                assert metric in divergence_pointwise_estimates_funcs, \
                        f"Unknown metric {metric}. " \
                        f"Options are: {list(divergence_pointwise_estimates_funcs.keys())}"
                self.divergence_estimates_proposal_base[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
                self.divergence_estimates_policy_base[metric] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])

        self.features_moments_proposal = dict()
        self.features_moments_policy = dict()
        for (label, feature) in self.features:
            self.features_moments_proposal[label] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
            self.features_moments_policy[label] = WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"])
        if self.features:
            self.eval_samples_updated.enroll(self._update_features_moments)

    def _update_features_moments(self, context, samples, proposal_log_scores, policy_log_scores, target_log_scores):
        """
        Improves the importance sampling estimates of the feature moments
        specified on construction of the Tuner

        Parameters
        ----------
        context: text
            context for the samples
        samples: list of items
            samples from the proposal model
        proposal_log_scores: array of floats
            log-probabilities for the samples according to the proposal
        policy_log_scores: array of floats
            log-probabilities for the samples according to the policy
        target_log_scores: array of floats
            log-probabilities for the samples according to the target
        """
        device = get_device(proposal_log_scores)
        policy_log_scores = policy_log_scores.to(device)
        logweights = policy_log_scores - proposal_log_scores
        importance_ratios = torch.exp(logweights)
        for (label, feature) in self.features:
            proposal_moment_pointwise_estimates = feature.log_score(samples, context=context).exp().to(device)
            self.features_moments_proposal[label].update(proposal_moment_pointwise_estimates)
            self.features_moments_policy[label].update(importance_ratios * proposal_moment_pointwise_estimates)

    def _update_moving_z(self, contexts, proposal_log_scores, target_log_scores):
        target_log_scores, proposal_log_scores = to_same_device(target_log_scores, proposal_log_scores)
        z_pointwise_estimates_batch = torch.exp(target_log_scores - proposal_log_scores)
        for i, context in enumerate(contexts):
            self.z[context].update(z_pointwise_estimates_batch[i])
            self.metric_updated.dispatch('z', self.z[context].value)

    def _update_divergence_estimates(self, estimates_dict, contexts, p_log_scores, q_log_scores, use_z=False, proposal_log_scores=None):
        p_log_scores, q_log_scores = to_same_device(p_log_scores, q_log_scores)
        if proposal_log_scores is not None:
            proposal_log_scores = proposal_log_scores.to(p_log_scores.device)

        z_values = torch.tensor([self.z[c].value if use_z else 1.0 for c in contexts], device=p_log_scores.device)
        valid_mask = z_values > 0

        if not torch.any(valid_mask):
            return

        for div_type, estimator in estimates_dict.items():
            pointwise_estimates = divergence_pointwise_estimates_funcs[div_type](
                p_log_scores[valid_mask],
                q_log_scores[valid_mask],
                z_values[valid_mask].unsqueeze(1),
                proposal_log_scores=proposal_log_scores[valid_mask] if proposal_log_scores is not None else None
            )
            estimator.update(pointwise_estimates.flatten())

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

    def _report_all_importance_sampling_estimates(self):
        """
        Reports all tracked metrics for a given context
        """
        self._report_importance_sampling_estimates(self.divergence_estimates_target_policy, 'target_policy')
        self._report_importance_sampling_estimates(self.divergence_estimates_target_proposal, 'target_proposal')
        if self.track_divergence_from_base:
            self._report_importance_sampling_estimates(self.divergence_estimates_policy_base, 'policy_base')
            self._report_importance_sampling_estimates(self.divergence_estimates_proposal_base, 'proposal_base')
        self._report_importance_sampling_estimates(self.features_moments_proposal, 'proposal')
        self._report_importance_sampling_estimates(self.features_moments_policy, 'policy')

    def _update_proposal_if_better(self):
        """
            Checks if D(p||.) is lower for policy than for the proposal
            and if so, updates the proposal
        """
        divergence_target_proposal = average(self.divergence_estimates_target_proposal[self.params["proposal_update_metric"]])
        divergence_target_policy = average(self.divergence_estimates_target_policy[self.params["proposal_update_metric"]])
        if divergence_target_proposal > \
                divergence_target_policy:
            self.proposal.model.load_state_dict(self.policy.model.state_dict())
            self.metric_updated.dispatch('proposal_updated', 1)
            self.proposal_updated.dispatch(self.proposal, self.params['proposal_update_metric'],
                                            divergence_target_policy, divergence_target_proposal)
        else:
            self.metric_updated.dispatch('proposal_updated', 0)

    def _compute_gradient_batched(self, samples_nested, proposal_log_scores, target_log_scores, policy_log_scores, contexts, n_steps):
        """
        Computes the gradient on a minibatch of samples across all contexts.
        """
        z_values = [self.z[c].value for c in contexts]

        losses = []
        for i, context in enumerate(contexts):
            z_value = z_values[i]
            if z_value > 0:
                z = torch.tensor(z_value, device=policy_log_scores.device, dtype=policy_log_scores.dtype)
                loss = self._loss(
                    samples_nested[i], context, proposal_log_scores[i],
                    target_log_scores[i], policy_log_scores[i], z
                )
                losses.append(loss)

        if not losses:
            return

        # Aggregate losses and perform a single backward pass
        total_loss = torch.stack(losses).mean()
        scaled_loss = total_loss / n_steps

        self.metric_updated.dispatch('loss', scaled_loss.item())
        scaled_loss.backward()

    def _step(self):
        """
        Performs a tuning step of the policy distribution's model

        Performs a single step of gradient updates on a batch of samples:
          - obtains samples and their log-scores from the proposal model
          - repeats gradient computations, with minibatches
          - applies the accumulated gradients

        """
        contexts = self._get_next_context_batch()
        num_contexts = len(contexts)
        sampler = AccumulationSampler(self.proposal, total_size=self.params["n_samples_per_context"])

        with Timer() as t_sampling:
            samples_nested, proposal_log_scores = sampler.sample_batch(
                contexts=contexts,
                sampling_size=self.params["sampling_size"]
            )
            samples_flat = [s for sublist in samples_nested for s in sublist]
        self.metric_updated.dispatch("timing/generation_per_sample", t_sampling.elapsed / len(samples_flat))

        scoring_size = self.params["scoring_size"]

        with Timer() as t_scoring:
            target_log_scores = score_in_chunks_batched(
                self.target, samples_nested, contexts, scoring_size
            )
        self.metric_updated.dispatch("timing/target_scoring_per_sample", t_scoring.elapsed / len(samples_flat))

        if self.track_divergence_from_base:
            base = self.target.scorers[0]
            base_log_scores = score_in_chunks_batched(
                base, samples_nested, contexts, scoring_size
            )

        self.proposal.report_samples_stats(samples_flat, contexts, self.metric_updated)
        self._update_moving_z(contexts, proposal_log_scores, target_log_scores)
        self._update_divergence_estimates(self.divergence_estimates_target_proposal, contexts, target_log_scores, proposal_log_scores, use_z=True)
        if self.track_divergence_from_base:
             self._update_divergence_estimates(self.divergence_estimates_proposal_base, contexts, proposal_log_scores, base_log_scores, use_z=False, proposal_log_scores=proposal_log_scores)

        n_samples_to_boost, n_samples_to_downweight = 0, 0

        all_indices = [
            (i, j)
            for i in range(len(contexts))
            for j in range(self.params["n_samples_per_context"])
        ]
        total_samples = len(all_indices)

        # Determine the number of steps based on the total sample count
        n_steps = (total_samples + scoring_size - 1) // scoring_size

        for s in trange(n_steps, desc="mini-steps", leave=False):
            self.ministep_idx_updated.dispatch(s)

            # Create the minibatch from the flat index
            start_idx = s * scoring_size
            end_idx = min(start_idx + scoring_size, total_samples)
            minibatch_indices = all_indices[start_idx:end_idx]

            # Group sample indices by their original context
            indices_by_context = defaultdict(list)
            for ctx_idx, sample_idx in minibatch_indices:
                indices_by_context[ctx_idx].append(sample_idx)

            mb_context_indices = sorted(indices_by_context.keys())
            mb_contexts = [contexts[i] for i in mb_context_indices]

            # Each item in these lists corresponds to a context in `mb_contexts`.
            mb_samples_nested = []
            mb_proposal_log_scores_ragged = []
            mb_target_log_scores_ragged = []
            if self.track_divergence_from_base:
                mb_base_log_scores_ragged = []

            for i in mb_context_indices:
                sample_indices_for_ctx = indices_by_context[i]

                # Gather samples
                mb_samples_nested.append([samples_nested[i][j] for j in sample_indices_for_ctx])

                # Gather scores using advanced tensor indexing
                idx_tensor = torch.LongTensor(sample_indices_for_ctx).to(proposal_log_scores.device)
                mb_proposal_log_scores_ragged.append(proposal_log_scores[i, idx_tensor])
                mb_target_log_scores_ragged.append(target_log_scores[i, idx_tensor])
                if self.track_divergence_from_base:
                    mb_base_log_scores_ragged.append(base_log_scores[i, idx_tensor])

            mb_policy_log_scores_ragged = self.policy.log_score_batch(
                samples=mb_samples_nested,
                contexts=mb_contexts,
                grad=True
            )
            detached_policy_scores_ragged = [t.detach() for t in mb_policy_log_scores_ragged]

            # These calls now process one context from the minibatch at a time.
            for i, context in enumerate(mb_contexts):
                self._update_divergence_estimates(
                    self.divergence_estimates_target_policy, [context],
                    mb_target_log_scores_ragged[i].unsqueeze(0),
                    detached_policy_scores_ragged[i].unsqueeze(0),
                    use_z=True,
                    proposal_log_scores=mb_proposal_log_scores_ragged[i].unsqueeze(0)
                )
                if self.track_divergence_from_base:
                    self._update_divergence_estimates(
                        self.divergence_estimates_policy_base, [context],
                        detached_policy_scores_ragged[i].unsqueeze(0),
                        mb_base_log_scores_ragged[i].unsqueeze(0),
                        use_z=False,
                        proposal_log_scores=mb_proposal_log_scores_ragged[i].unsqueeze(0)
                    )

            for i, context in enumerate(mb_contexts):
                self.eval_samples_updated.dispatch(
                    context, mb_samples_nested[i], mb_proposal_log_scores_ragged[i],
                    detached_policy_scores_ragged[i], mb_target_log_scores_ragged[i]
                )

            with Timer() as t_backprop:
                # NOTE: Assumes `_compute_gradient_batched` is adapted to handle ragged lists of tensors.
                self._compute_gradient_batched(
                    mb_samples_nested, mb_proposal_log_scores_ragged, mb_target_log_scores_ragged,
                    mb_policy_log_scores_ragged, mb_contexts, n_steps
                )
            # Use the actual number of samples in the minibatch for accurate timing
            self.metric_updated.dispatch("timing/backpropagation_per_sample", t_backprop.elapsed / len(minibatch_indices))

            # Flatten the ragged lists to perform vectorized calculations across the entire minibatch.
            mb_target_flat = torch.cat(mb_target_log_scores_ragged)
            detached_policy_flat = torch.cat(detached_policy_scores_ragged)

            # Create a corresponding flat tensor of z-values for each sample.
            z_values_list = []
            for i, ctx_idx in enumerate(mb_context_indices):
                n_samples_for_ctx = len(mb_samples_nested[i])
                z_val = self.z[contexts[ctx_idx]].value
                z_values_list.extend([z_val] * n_samples_for_ctx)
            z_values_tensor_flat = torch.tensor(z_values_list, device=mb_target_flat.device)

            in_support = ~mb_target_flat.isneginf()
            valid_z = z_values_tensor_flat > 0
            valid_mask = in_support & valid_z

            if torch.any(valid_mask):
                norm_target_scores = mb_target_flat[valid_mask] - torch.log(z_values_tensor_flat[valid_mask])
                policy_scores_valid = detached_policy_flat[valid_mask]
            n_samples_to_boost += (norm_target_scores > policy_scores_valid).sum().item()
            n_samples_to_downweight += (norm_target_scores < policy_scores_valid).sum().item()

        # Report final metrics after all mini-batches
        self._report_all_importance_sampling_estimates()

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
        Fine-tunes policy distribution's model

        Fine-tunes the model of the policy distribution:
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
        val_policy = self.policy.validation()
        features_moments = {label: WindowedMovingAverage(window_size=self.params["estimates_rolling_window_size"]) for label, _ in self.features}

        for val_contexts in tqdm(val_dataloader, desc="validation", leave=False):
            # We sample one per context
            samples, _ = val_policy.sample_batch(val_contexts, sampling_size=1)

            for (label, feature) in self.features:
                for context, context_samples in zip(val_contexts, samples):
                    pointwise_estimates = feature.log_score(context_samples, context=context).exp()
                    features_moments[label].update(pointwise_estimates)

        for (label, _) in self.features:
            self.metric_updated.dispatch(f"val_{label}", features_moments[label].value)