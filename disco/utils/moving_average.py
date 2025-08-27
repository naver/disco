# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
import torch

def average(moving_averages):
    if moving_averages.items():
        weighted_values, weights = list(zip(
                *[(ma.value * ma.weight, ma.weight) for _, ma in moving_averages.items()]
            ))
        return sum(weighted_values) / sum(weights)
    else:
        return 0

class WindowedMovingAverage:
    """
    Keeps a moving average of a quantity over a fixed-size window.
    """
    def __init__(self, window_size=1000):
        """
        Parameters
        ----------
        window_size: int
            The number of samples to average over.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.buffer = []
        self.value = None  # Stores the average of the last completed window.

    def update(self, new):
        """
        Adds new values to the buffer and updates the average

        Parameters
        ----------
        new: torch.Tensor, np.ndarray, list, or float
            A collection of new pointwise estimates to add to the buffer.
        """
        # Ensure 'new' is a flat list of numbers
        if isinstance(new, torch.Tensor):
            new_values = new.flatten().tolist()
        elif isinstance(new, np.ndarray):
            new_values = new.flatten().tolist()
        elif hasattr(new, '__iter__'):
            new_values = list(new)
        else:
            new_values = [new]  # Treat a single number as a list

        self.buffer.extend(new_values)

        # crop to window size
        self.buffer = self.buffer[-self.window_size:]

        # Calculate the mean of the full window and update the public value
        self.value = sum(self.buffer) / len(self.buffer)

class MovingAverage:
    """
    Keeps a moving average of a quantity.
    The average is initialized with the first value reported.
    """
    def __init__(self):
        """
        Initializes the moving average.
        """
        self.value = None
        self.weight = 0

    def update(self, new):
        """
        Adds new values to the average.

        Parameters
        ----------
        new: torch.Tensor, np.ndarray, list, or float
            A collection of new pointwise estimates to add.
        """
        # Ensure 'new' is a flat list of numbers
        if isinstance(new, torch.Tensor):
            new_values = new.detach().flatten().tolist()
        elif isinstance(new, np.ndarray):
            new_values = new.flatten().tolist()
        elif hasattr(new, '__iter__'):
            new_values = list(new)
        else:
            new_values = [new]  # Treat a single number as a list

        if not new_values:
            return

        new_weight = len(new_values)
        new_mean = sum(new_values) / new_weight

        # If this is the first update, initialize the moving average
        if self.value is None:
            self.value = new_mean
            self.weight = new_weight
        else:
            # Update the existing moving average incrementally
            total_weight = self.weight + new_weight
            self.value = (self.value * self.weight + new_mean * new_weight) / total_weight
            self.weight = total_weight