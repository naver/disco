# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import tqdm.autonotebook as tqdm

def batchify(func, batch, samples=list(), **args):
    all = []
    with tqdm.tqdm(total=len(samples), desc=func.__name__, position=1, leave=False) as pbar:
        for i in range(len(samples)//batch + 1):
            subsamples = samples[i * batch:(i+1) * batch]
            if subsamples:
                all.append(func(subsamples, **args))
            pbar.update(batch)
    return torch.cat(all)

def score_in_chunks_batched(model, samples_nested, contexts, chunk_size):
    """
    Scores samples from multiple contexts in chunks to manage memory, generalized
    to handle any chunk size.

    If the `chunk_size` is smaller than the number of contexts, contexts are
    processed sequentially to avoid memory overflow.

    Parameters:
        model: The distribution model with a `log_score_batch` method.
        samples_nested (list of lists): Samples, structured as [context][sample_index].
        contexts (list of str): The list of contexts.
        chunk_size (int): The maximum number of (sample, context) pairs to score in a single pass.

    Returns:
        torch.Tensor: A tensor of scores with shape (num_contexts, total_samples).
    """
    num_contexts = len(contexts)
    # Handle empty inputs to prevent errors
    if num_contexts == 0:
        return torch.empty(0, 0)

    num_samples_per_context = len(samples_nested[0])
    if num_samples_per_context == 0:
        return torch.empty(num_contexts, 0)

    desc = f"Scoring ({model.__class__.__name__})"

    # --- Case 1: `chunk_size` is large enough for all contexts at once ---
    if chunk_size >= num_contexts:
        all_scores_chunks = []
        # Calculate how many samples PER CONTEXT can fit in one batch.
        samples_per_context_chunk = max(1, chunk_size // num_contexts)

        for i in tqdm.trange(0, num_samples_per_context, samples_per_context_chunk, desc=desc, leave=False):
            chunk_slice = slice(i, i + samples_per_context_chunk)
            chunk_samples_nested = [sublist[chunk_slice] for sublist in samples_nested]

            if not chunk_samples_nested[0]:
                continue

            # Score this chunk across all contexts in one parallel call
            scores_chunk = model.log_score_batch(
                samples=chunk_samples_nested,
                contexts=contexts,
            )  # Shape: (num_contexts, samples_per_context_chunk)
            all_scores_chunks.append(scores_chunk)

        return torch.cat(all_scores_chunks, dim=1)

    # --- Case 2: `chunk_size` is smaller than num_contexts ---
    else:
        all_context_scores = []
        for i in tqdm.trange(num_contexts, desc=desc, leave=False):
            # Isolate samples and context for the current iteration
            current_context = [contexts[i]]
            current_samples_list = samples_nested[i]

            scores_for_one_context = []
            # For this single context, we can use the full `chunk_size` for its samples.
            for j in range(0, num_samples_per_context, chunk_size):
                sample_chunk = [current_samples_list[j : j + chunk_size]]

                if not sample_chunk[0]:
                    continue

                scores_chunk = model.log_score_batch(
                    samples=sample_chunk,
                    contexts=current_context,
                ) # Shape: (1, len_of_chunk)
                scores_for_one_context.append(scores_chunk)

            # After processing all sample chunks, concatenate them for the current context
            if scores_for_one_context:
                full_scores = torch.cat(scores_for_one_context, dim=1)
                all_context_scores.append(full_scores)

        # Finally, concatenate the results from all contexts
        return torch.cat(all_context_scores, dim=0)

def get_token_first_indices(x, token):
    """Find the first occurrence of a token in a 2D token array.

    Parameters
    ----------
    x: 2D int array
       list of token sequences
    token: int
        token to search

    Returns
    ------
    1D array containing the position of the first occurrence of the token or -1 if not found
    """
    if 0 == x.shape[-1]:
        return torch.tensor(-1).repeat(x.shape[0])
    else:
        mask = token == x
        mask_max_values, mask_max_indices = torch.max(mask, dim=1)
        mask_max_indices[mask_max_values == 0] = -1
        return mask_max_indices
