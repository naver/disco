# 🕺🏽 disco: A Toolkit for Distributional Control of Generative Models

The 🕺🏽 **disco** toolkit allows to control language models and other generative systems to match human preferences while avoiding catastrophic forgetting.

To achieve this, **disco** decouples the problem of expressing _what_ properties the model should have from _how_ to actually get the desired model as separate steps.

**Step 1: ⚓ We express how the target distribution *should* be**

First, we define some feature over the generated samples that matters to us. It can be anything we can compute. For example, on a language model it can be as simple as whether the generated text contains a certain word or as complex as the compilability of some generated piece of code. Importantly, there is no need for the feature to be differentiable.

Then, we can express our preferences on the target distribution by deciding how prevalent the feature should be. For example, we might want to ask that a certain word appears 50% of the time when sampling from the model; or that 100% of the generated code is compilable. The resulting target distribution is expressed as an energy-based model or EBM, which is an unnormalized probability distribution that respects the desired moments while avoiding catastrophic forgetting, as a result of having minimal KL divergence to the original model.

The resulting representation of the target distribution can *score* samples, but cannot directly be used to *generate* them.

**Step 2: 🎯 Approximate the target distribution**

To generate samples from the target distribution we can tune a model to approximate it. We do this by minimizing the divergence to the target distribution. While techniques such as reinforcement learning from human feedback (RLHF) are restricted to using one kind of divergence only (specifically, reverse KL divergence), **disco** is more general, allowing the use of the full class of f-divergences, including both forward and reverse KL divergence, Jensen-Shannon, and total variation distance.

**Step 3: 💬 Generate content that matches the preferences**

The resulting model can generate samples directly from a close approximation of the target distribution. Furthermore, it can be used jointly with Quasi-Rejection Sampling (QRS), a Monte Carlo sampling technique that allows the generation of samples that are even more representative of the target distribution.
Alternatively, it is then possible to use decoding methods such as nucleus sampling, top-k sampling, or beam search, which would return samples from a further updated target distribution.

See the references below for more theoretical and technical details.

## Installation

### Standard installation

The easiest way to install **disco** is to rely on pip, asking for the ```disco-generation``` package:

```
pip install disco-generation
```

Note that the toolkit:
  - depends on PyTorch,
  - uses HuggingFace's Transformers library for the generic handling of language models, as well as the generation of samples from them.

 ### Toolkit Developers

If we plan to extend the toolkit we will need to clone and to install it as a local package.
From the toolkit top folder, once we've git-cloned the repository and activated our development environment, we simply do:
```
pip install -e .
```

## Quick introduction


### Distributions

The generative model that we want to tune must be wrapped by a `Distribution` object. For example, for a (causal or seq2seq) language model compatible with the 🤗 Hugging Face interface use an `LMDistribution`.

A valid `Distribution` must have the following two methods:
  - `.sample(context)` that given an optional `context` on which the distribution can be conditioned, returns a list of samples from the underlying distribution and a tensor with their corresponding log-probabilities;
  - `.log_score(samples, context)` that given a list of samples and the `context` on which to condition the distribution, returns their corresponding log-probabilities.

```python
from disco.distributions import LMDistribution
distribution = LMDistribution()

incipit = "It was a cold and stormy night"
samples, log_scores = distribution.sample(context=incipit)

distribution.log_score(samples, context=incipit)
```

`LMDistribution` generate samples, with the `TextSample` type, which are named tuples with both a `text` and `token_ids` fields.

<small>From now on, after this initial example, imports will be skipped for clarity.</small>

### Features

Features are represented by an object with the method

- `.score(samples, context)` which given a list of samples and an eventual context returns a tensor of real-valued scores.

A convenient way to define one is using the `Scorer` class, which accepts a function or a lambda abstraction that takes sample and a context, and vectorizes it. For example, we can compute the effective length of a GPT-2 text sample by finding the eos token:

```python
sequence_length = Scorer(lambda s, c: s.text.index("<|endoftext|>"))
```

Where `s` is the sample (assumed to be a `TextSample`) and `c` is an eventual context.

#### Boolean Features

An important class of features are *boolean* features. While general features can only be used to define *distributional* constraints, boolean features can also be used to define *pointwise* constraints, see below. To define one, we can use the `BooleanScorer` helper class, which takes a function as an argument. For example, we can score the presence of the string "amazing", as follows:

```python
amazing = BooleanScorer(lambda s, c: "amazing" in s.text)
```

The ```False```/```True``` results from the lambda are casted to `0.0`/`1.0` float values so that they can be used in the EBM definition.

`BooleanScorer` belongs to the more general `PositiveScorer` class, which can be used to construct EBMs. The main properties of a `PostiveScorer` are that first, it returns positive scorers, and second that it provides the method:

 - `.log_score(samples, context)` that given a list of samples and the `context` on which to condition the distribution, returns their corresponding log-probabilities.

As a consequence, we can see that a ```Distribution``` is also a ```PositiveScorer``` that is able to sample as well.


### Controlling Generation

#### Expressing preferences through an EBM

We express preferences over the distribution by defining target moments for specific features. This results in a target distribution that matches the desired moments while minimizing the KL divergence to the original distribution. In other words, it incorporates the preferences while avoiding catastrophic forgetting. This distribution is represented as an EBM, which can be used to score samples, in other words it is a `PositiveScorer`, but cannot be used to sample, we'll see how to sample below.

We can express either *pointwise* or *distributional* constraints on a distribution and compose them at will. The former expresses a (boolean) property that must apply to *all* sequences, whereas the latter represents properties at the distributional level.

To obtain the target distribution that incorporates our constraints, we use the `constraint` method of the corresponding `Distribution`. This method takes a list of features and their corresponding target moments.

For example, we can define an EBM with a *pointwise* constraint requiring that all our samples must include "amazing" by setting the target moment to `1` on a `BooleanFeature`:

```python
target_ebm = base.constrain([amazing], [1])
```

Or we can ask for a _distributional_ constraint requiring that _half_ of the samples include "amazing":

```python
target_ebm = base.constrain([amazing], [1/2])
```


#### Approximating the target EBM


Given an EBM target distribution, we now want to train a model to approximate it so that we can use it to generate samples. In the _unconditional_ case, namely when there is a single fixed context used in generation, then we can use a `Tuner`, more specifically a ```DPGTuner```, as follows.


```python
target_ebm = base.constrain([amazing], [1])

model = LMDistribution(freeze=False)
incipit = "It was a cold and stormy night"

tuner = DPGTuner(model, target_ebm, context=incipit)
tuner.tune()
```

And we can sample _amazing_ sequences from the tuned model.
```python
samples, log_scores = model.sample(context=incipit)
for s in samples:
  print(incipit + s.text)
```

##### Tuning parameters

Important parameters of the `Tuner` include:

- `n_gradient_steps`: number of total gradient steps in the full tuning process;
- `n_samples_per_step`: total number of samples used in performing a gradient step (aka batch size);
- `scoring_size`: number of samples sent in a batch to the `.score` function. This parameter affects training speed or helps solve GPU memory errors, but does not affect final results;
- `sampling_size`: number of samples obtained from a single call to the `.sample` function. This parameter affects training speed or helps solve GPU memory errors, but does not affect final results;
- `features`: list of pairs (`name`, `feature`) so that the `feature` moments will be computed by importance sampling (and reported using the key given by `name`);
- `track_divergence_from_base`: set to True to track the reverse KL divergence from the original model —this requires an additional round of samples' scoring).

#### Logging

The Tuner reports a number of metrics that are useful to monitor the training progress. A number of `Logger` classes are provided to keep track of these metrics. Basic logging is provided though the console, as follows:

```python
console_logger = ConsoleLogger(tuner)
```

However, more detailed statistics can be kept trhough a JSON/WandB/Neptune loggers:

```python
project = "example_project"
name = "run_01"
json_logger = JSONLogger(tuner, project, name)
neptune_logger = NeptuneLogger(tuner, project, name)
wandb_logger = WandBLogger(tuner, project, name)
```

where `project` and `name` refer to the project and run name, respectively.

##### Logged Metrics

Loggers store a number of metrics about the training process. Here we list a few of the most relevant ones:

-  `kl_target_model` and `kl_target_proposal`: estimates of the forward KL divergence to the target EBM from the tuned model and the proposal distribution, respectively. In the case of using online training, the two are equivalent with the only caveat that `kl_target_model` is computed —this is the metric being optimized, and not the value reported as `loss`;
-  `kl_model_base` and `kl_proposal_base`: estimates of the reverse KL divergence to the original model of the tuned model and the proposal distribution, respectively —only reported if `track_divergence_from_base` is set to True;
-  Feature moments: Estimate of the features' moments for those features specified with the `features` parameter at the Tuner's construction time.

### Controlled Conditional Generation

The _conditional_ case is superficially very similar, with an extra step needed to instantiate a `ContextDistribution`, which allows to sample contexts that can then be used to condition the model. Furthermore, we use the more general ```CDPGTuner``` class.

Assuming we have a file of incipits, one per line, in a `data/incipits.txt` file, we could do:

```python
target_ebm = base.constrain([amazing], [1])

model = LMDistribution(freeze=False)

tuner = CDPGTuner(model, target_ebm,
  context_distribution=ContextDistribution("data/incipits.txt"),
  context_sampling_size=2**3)
tuner.tune()
```

Note that while we have used a decoder-only model here for illustrative purposes, the real power of the CDPGTuner is that it allows to control _seq2seq models_ such as those used in NMT, summarization, etc... Please refer to the dedicated [tutorial notebook](tutorials/4.conditional_tuning.ipynb) for an example of how to control an actual conditional model.


### Improving the approximation through minimizing other f-divergences

The Tuner classes train the model by minimizing its divergence from the distribution induced by the target ebm. While in the original DPG and CDPG algorithms this divergence was always the KL divergence, [recent work](https://arxiv.org/abs/2302.08215) has generalized them to the wider class of [f-divergences](https://en.wikipedia.org/wiki/F-divergence). To pick the loss to minimize, use the corresponding `FDPGTuner` or `FCDPGTuner` class, depending on whether you are in the unconditional or conditional case, and choose the divergence to minimize through the `loss` parameter. Some of the possible losses are:

- `KLLoss()`: KL divergence
- `ReverseKLLoss()`: KL divergence reversing the order of the arguments
- `TVLoss()`: Total Variation Distance
- `JSLoss()`: Jensen-Shannon divergence.

Each of these divergences strikes a different balance between level of alignment and diversity. KL exhibits lower alignment than other losses, but higher diversity. On the other hand, ReverseKL tends to produce hight alignment at the cost of lower diversity. Jensen-Shannon strikes a good balance between the two, making it a good default choice.

As an example,

```python
target_ebm = base.constrain([amazing], [1])

model = LMDistribution(freeze=False)

tuner = FCDPGTuner(model, target_ebm, loss=JSLoss(),
  context_distribution=ContextDistribution("data/incipits.txt"),
  context_sampling_size=2**3)
tuner.tune()
```

### Reinforcement Learning from Human Feedback (RLHF)

RLHF is a popular paradigm for aligning language models to preferences. While RLHF is commonly known in the form of a reward maximization algorithm, [recent work](https://arxiv.org/abs/2206.00761) has shown that it is equivalent to a distribution approximation problem which can be easily handled by **disco**. Specifically, given a reward function `r(x)` and the regularization parameter `beta`, the following code optimizes the same objective as RLHF:

```python
target_ebm = base * ExponentialScorer([r], [1./beta])

model = base.clone()


tuner = FCDPGTuner(model, target_ebm, loss=ReverseKLLoss())
```

In other words, RLHF optimizes the *reverse* KL to the above-defined target EBM. Interestingly, this opens new opportunities as the divergence to be minimized could be now chosen to be any other, as explored [here](https://arxiv.org/abs/2302.08215).


#### Monte-Carlo sampling to improve the approximation

After the tuning is done, `model` is now a better approximation to the target EBM, but it is not guaranteed to perfectly match this distribution. While further training can improve the situation, another alternative is using [quasi-rejection sampling (QRS)](https://disco.europe.naverlabs.com/QRS/), a Monte-Carlo sampling technique that allows to trade-off sampling efficiency for a higher fidelity to the target distribution —a higher value of `beta` yields a better fidelity although at a higher computational cost.

```python
beta=0.5
sampler = QuasiRejectionSampler(target_ebm, model, beta=beta)
samples, log_scores = sampler.sample(sampling_size=2**7)
```

#### In summary

To put some of this (distributional constraint, tuning in the unconditional case and using QRS) together:

```python
base = LMDistribution()
target_ebm = base.constrain([amazing], [1/2])

model = LMDistribution(freeze=False)

tuner = DPGTuner(model, target_ebm)
tuner.tune()

beta=0.5
sampler = QuasiRejectionSampler(target_ebm, model, beta=beta)
samples, log_scores = sampler.sample(context=incipit, sampling_size=2**7)
```

### Going further

A few things to keep in mind while reading the following paragraphs showing the principles of **disco**:
  1. this is only an introduction, skipping some details and relying on toyish use cases;
  1. the notebooks in the tutorials folder go in more depth, on more use cases;
  1. the focus here and in most notebooks is on natural language, but again the toolkit can be used to control distributions over sequences such as code or chess moves, or even other data types, as long as they respect the basic assumptions of a disco `Distribution` object.

## References

The **disco** toolkit implements the theoretical framework presented in the following works:
  - A Distributional Approach to Controlled Text Generation, Khalifa et al., 2021, <https://openreview.net/forum?id=jWkw45-9AbL>, ICLR;
  - An approximate sampler for energy-based models with divergence diagnostics, Eikema et al., 2022, <https://openreview.net/forum?id=VW4IrC0n0M>, TMLR;
  - Energy-Based Models for Code Generation under Compilability Constraints, Korbak et al., 2021, <https://arxiv.org/abs/2106.04985>, ACL (Workshop on Natural Language Processing for Programming);
  - Controlling Conditional Language Models without Catastrophic Forgetting, Korbak et al., 2022, <https://proceedings.mlr.press/v162/korbak22a.html>, ICML;
  - On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting, Korbak et al., 2022, <https://openreview.net/forum?id=XvI6h-s4un>, NeurIPS;
  - Aligning Language Models with Preferences through f-divergence Minimization, Go et al., 2023, https://arxiv.org/abs/2302.08215, ICML.

To cite **disco**, please use:
```
@inproceedings{kruszewski-etal-2023-disco,
    title = "disco: a toolkit for Distributional Control of Generative Models",
    author = "Kruszewski, Germ{\'a}n  and
      Rozen, Jos  and
      Dymetman, Marc",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.14",
    pages = "144--160",
    abstract = "Pre-trained language models and other generative models have revolutionized NLP and beyond. However, these models tend to reproduce undesirable biases present in their training data. Also, they may overlook patterns that are important but challenging to capture. To address these limitations, researchers have introduced distributional control techniques. These techniques, not limited to language, allow controlling the prevalence (i.e. expectations) of any features of interest in the model{'}s outputs. Despite their potential, the widespread adoption of these techniques has been hindered by the difficulty in adapting the complex, disconnected code. Here, we present disco, an open-source Python library that brings these techniques to the broader public",
}
```

## License

See [LICENSE](LICENSE) file.
