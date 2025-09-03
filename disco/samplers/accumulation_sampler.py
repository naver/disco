# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import Sampler
import torch
from tqdm.autonotebook import trange


class AccumulationSampler(Sampler):
    """
    Utility class to accumulate samples, up to a total size
    """

    def __init__(self, distribution, total_size=512):
        """
        Parameters
        ----------
        distribution: distribution
            distribution to sample from
        total_size: int
            total number of samples
        """

        self.distribution = distribution
        self.total_size = total_size

    def sample(self, sampling_size=32, context=""):
        """accumulates batches of samples from the distribution

        Parameters
        ----------
        sampling_size: int
            number of requested samples per individual sampling
        context: text
            contextual text for which to sample

        Returns
        -------
        a tuple of accumulated samples and scores
        """
        with trange(
                self.total_size,
                desc=f"sampling from {type(self.distribution).__name__}",
                position=1,
                leave=False
            ) as t:
            remaining = self.total_size
            samples, log_scores = list(), torch.empty([0])
            while remaining > 0:
                more_samples, more_log_scores = self.distribution.sample(context=context, sampling_size=sampling_size)
                length = min(remaining, len(more_samples))
                more_samples, more_log_scores = more_samples[:length], more_log_scores[:length]
                samples, log_scores = (
                        samples + more_samples,
                        torch.cat((log_scores, more_log_scores))
                    ) if samples else (more_samples, more_log_scores)
                remaining -= len(more_samples)
                t.update(len(more_samples))

        return (samples, log_scores)

    def sample_batch(self, contexts, sampling_size=32, output_scores=True):
        """
        Accumulates batches of samples for a list of contexts simultaneously.

        This method repeatedly calls the distribution's `sample_batch` method
        until `total_size` samples are collected for each context.

        Parameters
        ----------
        contexts: list of str
            A list of contextual texts for which to sample.
        sampling_size: int
            The number of samples to request for each context per batch call.

        Returns
        -------
        tuple of (list of lists, torch.Tensor)
            - A list of lists, where the outer list corresponds to the input
              contexts and each inner list contains `total_size` sample objects.
            - A tensor of log scores with shape `(num_contexts, total_size)`.
        """
        if not isinstance(contexts, list) or not contexts:
            raise ValueError("contexts must be a non-empty list of strings.")

        num_contexts = len(contexts)

        # Initialize containers for accumulating results for each context
        accumulated_samples = [[] for _ in range(num_contexts)]
        if output_scores:
            # Store score tensors from each batch call in a list for each context
            accumulated_scores_parts = [[] for _ in range(num_contexts)]

        with trange(
            self.total_size * num_contexts,
            desc=f"Batch sampling for {num_contexts} contexts from {type(self.distribution).__name__}",
            position=1,
            leave=False
        ) as t:
            collected_count = 0
            while collected_count < self.total_size:
                # Determine how many more samples are needed to reach the goal.
                # This prevents over-sampling in the final iteration.
                remaining = self.total_size - collected_count
                current_batch_size = min(remaining, sampling_size)

                # Call the batched sampling method on the distribution
                ret = self.distribution.sample_batch(
                    contexts=contexts,
                    sampling_size=current_batch_size,
                    output_scores=output_scores
                )
                if output_scores:
                    more_samples_nested, more_log_scores = ret
                else:
                    more_samples_nested = ret
                # `more_log_scores` has shape: (num_contexts, current_batch_size)

                # Distribute the flat list of samples and batched scores to their
                # respective context accumulators.
                for i in range(num_contexts):
                    start_idx = i * current_batch_size
                    end_idx = start_idx + current_batch_size

                    # Add samples for the i-th context
                    accumulated_samples[i].extend(more_samples_nested[i])

                    if output_scores:
                        # Append the score tensor for the i-th context
                        accumulated_scores_parts[i].append(more_log_scores[i])

                collected_count += current_batch_size
                t.update(current_batch_size * num_contexts)

        if output_scores:
            # Finalize the scores by concatenating the collected tensor parts for each context
            # and then stacking them into a single (num_contexts, total_size) tensor.
            final_scores = torch.stack(
                [torch.cat(parts) for parts in accumulated_scores_parts],
                dim=0
            )

        if output_scores:
            return (accumulated_samples, final_scores)
        else:
            return accumulated_samples
