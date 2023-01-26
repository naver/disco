# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from abc import ABC, abstractmethod

class Sampler(ABC):
    """
    Top-level abstract class for all samplers
    """
    
    def __init__(self, target, proposal):
        self.proposal = proposal
        self.target = target

    @abstractmethod
    def sample(self):
        pass