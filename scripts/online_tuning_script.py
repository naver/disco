# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, re
from datetime import datetime
import torch

from disco.distributions import LMDistribution
from disco.scorers import BooleanScorer
from disco.tuners import DPGTuner
from disco.tuners.loggers.console import ConsoleLogger


word = "amazing"
incipit = "It was a cold and stormy night"
path = "models"
gpt = "gpt2" # or a larger gpt2-medium or another CLM from Transformers
dev0, dev1 = "cpu", "cpu" # "0", "1" to use GPUs
n_gradient_steps = 10 # 1000 or more for actual tuning
divergence_evaluation_interval = 2**2 # 2**4 for actual tuning?

base = LMDistribution(model=gpt, device=dev0)
has_word = lambda s, c: bool(re.search(f"\\b{word}\\b", s.text))
word_scorer = BooleanScorer(has_word)
target = base * word_scorer

model = LMDistribution(model=gpt, freeze=False, device=dev1)

tuner = DPGTuner(model, target,
        context = incipit,
        features = [(word, word_scorer)],
        n_gradient_steps=n_gradient_steps,
        n_samples_per_context=2**8,
        sampling_size=2**5,
        scoring_size=2**5,
        divergence_evaluation_interval=divergence_evaluation_interval,
        n_kl_samples=2**10)
console_logger = ConsoleLogger(tuner)
tuner.tune()

samples, _ = model.sample(context=incipit)
print("rate after tuning is:")
print(sum([has_word(s, _) for s in samples]) / len(samples))

stamp = datetime.now().strftime("%Y_%m_%d-%H:%M")
torch.save(model, os.path.join(path, f"{word}.{stamp}.pt"))
