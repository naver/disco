# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from disco.utils.observable import Observable

class BaseLoss:
    def __init__(self):
        self.metric_updated = Observable()

