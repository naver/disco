# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

def get_device(tensor):
    """Get the device of a tensor

    Parameters
    ----------
    tensor: Tensor
            tensor from which to extract the device

    Returns
    -------
    computation device (0-n or cpu)
    """

    device = tensor.get_device()
    return "cpu" if 0 > device else device


def to_same_device(*tensors, device=None):
    """Move all tensors to the same device (by default the first tensor's one)

    Parameters
    ----------
    tensors: Tensor
        tensors which we want to move to a common device

    Returns
    -------
    a tuple of tensors in the same order as the arguments moved to a common device
    """
    if device is None:
        device = get_device(tensors[0])
    return tuple(tensor.to(device) for tensor in tensors)
