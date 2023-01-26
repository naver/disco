# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch

class EMABaseline:
    def __init__(self, ema_weight=0.99):
        self.baseline = 0
        self.ema_weight = ema_weight

    def advantage(self, rewards):
        self.baseline = self.ema_weight * self.baseline + \
                (1 - self.ema_weight) * torch.mean(rewards)
        advantage = rewards - self.baseline
        return advantage
