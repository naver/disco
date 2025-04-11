# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import tqdm.autonotebook as tqdm

def batchify(func, batch, samples=list(), **args):
    all = []
    with tqdm.tqdm(total=len(samples), desc=func.__name__, position=1, leave=True) as pbar:
        for i in range(len(samples)//batch + 1):
            subsamples = samples[i * batch:(i+1) * batch]
            if subsamples:
                all.append(func(subsamples, **args))
            pbar.update(batch)
    return torch.cat(all)

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
