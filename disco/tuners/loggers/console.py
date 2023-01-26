# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from datetime import datetime
from .base import BaseTunerObserver
from tqdm.autonotebook import tqdm


class ConsoleLogger(BaseTunerObserver):
    def __init__(self, tuner):
        super(ConsoleLogger, self).__init__(tuner)
        self.ministeps_pbar = tqdm(total=tuner.params["n_samples_per_step"]//tuner.params["sampling_size"])

    def __exit__(self, *exc):
        stamp = datetime.now().strftime("%H:%M:%S (%Y/%m/%d)")
        print (f"finished at {stamp}")

    def on_parameters_updated(self, params):
        for k, v in params.items():
            print (f"{k}: ", v)

    def on_step_idx_updated(self, s):
        self.ministeps_pbar.reset()

    def on_ministep_idx_updated(self, s):
        self.ministeps_pbar.update(1)
