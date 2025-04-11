# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from setuptools import setup, find_packages

setup(
    name='disco-generation',
    version='1.1.1',
    description='A toolkit for distributional control of generative models',
    url='https://github.com/naver/disco',
    author='Naver Labs Europe', author_email='jos.rozen@naverlabs.com',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0',
    long_description="""The 🕺🏽 **disco** toolkit allows to control the properties of the generations by language models and other generative systems to match human preferences while avoiding catastrophic forgetting.

To achieve this in **disco**, we first represent in what ways we want to update original model as a target distribution and then, generate samples from this new distribution through a combination of learning or monte-carlo methods, as follows.

**Step 1: We express how the target distribution *should* be**

To have a handle on the generative model, we define some feature over the generated samples. It can be anything we can compute. For example, on a language model it can be as simple as whether the generated text contains a certain word or as complex as the compilability of some generated piece of code. Importantly, there is no need for the feature to be differentiable.  
Then, we can express our preferences on the target distribution by defining the target *moments* of this feature. For example, we might want to ask that a certain word appears 50% of the time when sampling from the model; or that 100% of the generated code is compilable. The resulting target distribution is expressed as an energy-based model or EBM, which is an unnormalized probability distribution that respects the desired moments while avoiding catastrophic forgetting, as a result of having minimal KL divergence to the original model.  
This representation of the target distribution can *score* samples, but cannot directly be used to *generate* them.

**Step 2: We generate samples from the target distribution**

To generate samples from the target distribution, if not perfectly, we can tune a model to approximate it. The resulting model can generate samples directly from a close approximation of the target distribution. Furthermore, it can be used jointly with Quasi-Rejection Sampling (QRS), a Monte Carlo sampling technique that allows the generation of samples that are even more representative of the target distribution.  
Alternatively, it is then possible to use decoding methods such as nucleus sampling, top-k sampling, or beam search, which would return samples from a further updated target distribution.""",
    long_description_content_type='text/markdown',
    packages=find_packages(include=['disco', 'disco.*']),
    python_requires='>=3.8',
    install_requires=['torch', 'transformers>=4.49',
            'numpy', 'scipy',
            'datasets', 'spacy',
            'notebook',
            'neptune-client', 'wandb'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
