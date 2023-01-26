# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np

def average(moving_averages):
    if moving_averages.items():
        weighted_values, weights = list(zip(
                *[(ma.value * ma.weight, ma.weight) for _, ma in moving_averages.items()]
            ))
        return sum(weighted_values) / sum(weights)
    else:
        return 0
    
class MovingAverage(object):
    """
    Keeps a moving average of some quantity
    """
    def __init__(self, init_value=0):
        """
        Parameters
        ----------
        init_value: float
            the initial value of the moving average
        """
        self.value = init_value
        self.weight = 0
        self.init_value = init_value

    def __iadd__(self, new_values):
        """
        Includes an array of values into the moving average

        Parameters
        ----------
        new_values: torch/np array of floats
            pointwise estimates of the quantity to estimate
        """
        self.add_array(new_values)
        return self
    
    def add_array(self, new_values):
        """
        Includes an array of values into the moving average

        Parameters
        ----------
        new_values: torch/np array of floats
            pointwise estimates of the quantity to estimate
        """
        new_value = new_values.mean()
        new_weight = len(new_values)
        self.add(new_value, new_weight)

    def add(self, new_value, new_weight=1):
        """
        Includes a single value into a moving average

        Parameters
        ----------
        new_value: float
            pointwise estimate of the quantity to estimate
        new_weight: float
            a weight by which to ponder the new value
        """
        self.value = (self.value * self.weight + new_weight * new_value) / \
                (self.weight + new_weight)
        self.weight += new_weight

    def reset(self):
        """
        Resets the moving average to its initial conditions
        """
        self.value = self.init_value
        self.weight = 0
