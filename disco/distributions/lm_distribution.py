# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import copy
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LogitsProcessorList, GenerationConfig

from collections import namedtuple

from .base_distribution import BaseDistribution
from disco.utils.helpers import get_token_first_indices


TextSample = namedtuple('TextSample', ['token_ids', 'text'])

class LMDistribution(BaseDistribution):
    """
    Language model distribution class, a core class for all NLP
    use-cases, relying on Huggingface's Transformers library.
    """

    def __init__(self,
            model="gpt2", tokenizer=None, auto=AutoModelForCausalLM, freeze=True,
            length=40, device="cpu",
            **config
        ):
        """
        Parameters
        ----------
        model: string
            Transformers' name of a causal or seq2seq language model
        tokenizer: string
            Transformers' name for the related tokenizer
        auto: class
            auto class from Transformers, default is AutoModelForCausalLM
            but AutoModelForSeq2SeqLM is also valid
        freeze: boolean
            flag to eventually (not) freeze the network's parameters
        length: int
            number of tokens in the samples
        device: string
            reference of the computing device
        config: kwarg
            parameters and values passed to transformers' ```generate(…)```
        """

        if isinstance(model, str):
            # assume also the tokenizer is a str
            self.tokenizer= AutoTokenizer.from_pretrained(tokenizer if tokenizer else model)
            assert auto in [AutoModelForCausalLM, AutoModelForSeq2SeqLM], "only AutoModel, AutoModelForCausalLM and AutoModelForSeq2SeqLM are valid options."
            self._load_network(auto, model, device)
        else:
            self.tokenizer = tokenizer
            self.network = model

        self.device = device
        self.network.to(self.device)
        self.network.eval() # to make sure scoring is consistent
        if freeze:
            self.freeze(True)

        self.length = length

        self.gen_config = GenerationConfig(**{
            "top_k": 0,
            "top_p": 1.0,
            "typical_p": 1.0,
            "temperature": 1.0,
            "num_beams": 1
        })
        self.gen_config.update(**config)

    def _load_network(self, auto, model, device):
        print(f"Loading model to {device}")
        t0 = time.time()
        self.network = auto.from_pretrained(model, device_map=device, trust_remote_code=True, use_safetensors=True)
        print(f"Model loaded in {time.time() - t0:.0f}s.", )
        if not self.network.config.is_encoder_decoder and self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.network.resize_token_embeddings(len(self.tokenizer))
            self.network.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def to(self, device):
        self.device = device
        self.network.to(self.device)

    def freeze(self, frozen=True):
        """Freeze (or unfreeze) parameters for gradient computation.

        Parameters
        ----------
        frozen: boolean (True)
            state to transition to, default is to freeze
        """

        self.network.requires_grad_(not frozen)

    def log_score(self, samples, context="", grad=False, sum=True):
        """Computes log-probabilities for the samples according
        to the language model network in the given context

        Parameters
        ----------
        samples: list(Sample)
            samples to (log-)score as a list()
        context: text
            context for which to (log-)score the samples
        grad: boolean
            flag to eventually compute the gradients, e.g. when fitting
        sum: boolean
            flag to eventually return token-level tensor of scores

        Returns
        -------
        tensor of log-probabilities
        """
        shapes = set([s.token_ids.shape for s in samples])
        assert 1 == len(shapes), f"sequences of token_ids should have the same shape, but got: {shapes}."

        if self.network.config.is_encoder_decoder:
            assert context is not None and context != "", "context (encoder input) is mandatory for encoder-decoder models"
        elif not context:
            context = self.tokenizer.bos_token

        tokenized_context = self.tokenizer([context] * len(samples), return_tensors="pt", add_special_tokens=True)
        tokenized_context["input_ids"] = tokenized_context["input_ids"].to(self.device)
        tokenized_context["attention_mask"] = tokenized_context["attention_mask"].to(self.device)

        tokenized_samples = dict()
        tokenized_samples["input_ids"] = torch.stack([sample.token_ids for sample in samples]).to(self.device)

        tokenized_samples = self._discount_padding_tokens(tokenized_samples)

        input_ids, encoder_input_ids, forward_kwargs, labels, prompt_length, last  = \
                self._get_forward_inputs(tokenized_context, tokenized_samples, samples)

        if grad:
            outputs = self.network(**forward_kwargs, labels=labels)
        else:
            with torch.no_grad():
                outputs = self.network(**forward_kwargs, labels=labels)

        all_logits = outputs.logits[:, prompt_length:last, :] # [n_samples, length, vocab]

        all_logits = self._process_and_warp_logits(all_logits, input_ids, encoder_input_ids)

        all_logprobs = all_logits.log_softmax(-1)

        seq_logprobs = torch.gather(
                all_logprobs, 2, tokenized_samples["input_ids"][:, :, None]
            ).squeeze(-1) # [n_samples, length]

        seq_logprobs = torch.where(1 == tokenized_samples["attention_mask"], seq_logprobs, torch.tensor(0.).to(self.device))

        return seq_logprobs.sum(dim=1) if sum else seq_logprobs

    def _discount_padding_tokens(self, tokenized_samples):
        first_eos_indices = get_token_first_indices(
                tokenized_samples["input_ids"],
                self.tokenizer.eos_token_id
            )
        tokenized_samples["attention_mask"] = torch.where(
                self.tokenizer.pad_token_id == tokenized_samples["input_ids"],
                0, 1
            )
        for i, ix in enumerate(first_eos_indices):
            if None != self.network.config.forced_bos_token_id and\
                self.network.config.forced_bos_token_id == tokenized_samples["input_ids"][i][0]:
                    tokenized_samples["attention_mask"][i][0] = 0
            else:
                tokenized_samples["attention_mask"][i][0] = 1  # at least score one token
            if ix != -1:  # if there is an pad token
                tokenized_samples["attention_mask"][i][ix] = 1  # score first pad token
                tokenized_samples["attention_mask"][i][ix + 1:] = 0  # ignore everything after it
        tokenized_samples["attention_mask"] = tokenized_samples["attention_mask"].to(self.device)
        return tokenized_samples


    def _get_forward_inputs(self, tokenized_context, tokenized_samples, samples):
        if self.network.config.is_encoder_decoder:
            prompt_length = None
            last = -1
            encoder_input_ids = tokenized_context["input_ids"]
            input_ids = tokenized_samples["input_ids"]
            forward_kwargs = {
                "input_ids": encoder_input_ids,
                "decoder_input_ids": input_ids,
            }
            input_ids, forward_kwargs = self.network._prepare_decoder_input_ids_for_generation(len(samples),
                "decoder_input_ids",
                forward_kwargs)
            forward_kwargs['decoder_input_ids'] = input_ids
            labels = forward_kwargs["decoder_input_ids"]
        else:
            prompt_length = tokenized_context["input_ids"].shape[-1] - 1
            last = -1
            encoder_input_ids = None
            input_ids = torch.cat((tokenized_context["input_ids"], tokenized_samples["input_ids"]), 1)
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.cat((tokenized_context["attention_mask"], tokenized_samples["attention_mask"]), 1)
            }
            labels = forward_kwargs["input_ids"]

        return input_ids, encoder_input_ids, forward_kwargs, labels, prompt_length, last


    def _process_and_warp_logits(self, all_logits, input_ids, encoder_input_ids):
        generation_config = copy.deepcopy(self.gen_config)
        self.network._prepare_special_tokens(generation_config, False, self.network.device)

        logits_processor = self.network._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList()
        )

        processed_logits = []
        for i in range(all_logits.size(1)):
            processed = logits_processor(input_ids[:, :i+1], all_logits[:, i, :])
            processed_logits.append(processed.unsqueeze(1))

        all_logits = torch.cat(processed_logits, dim=1)

        return all_logits

    def sample(self, context="", sampling_size=32, sum=True):
        """Samples sequences from the language model in the given context

        Parameters
        ----------
        context: text
            contextual text for which to sample
        sampling_size: int
            number of sequences to sample
        sum: Boolean
            flag to eventually return token-level tensor of scores

        Returns
        -------
        tuple of (list of Sample(tokens, text), tensor of logprobs)
        """
        if self.network.config.is_encoder_decoder:
            assert context is not None and context != "", "context (encoder input) is mandatory for encoder-decoder models"
        elif not context:
            context = self.tokenizer.bos_token

        tokenized_context = self.tokenizer(
            context,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        tokenized_contexts = {k: v.to(self.network.device).repeat(sampling_size, 1) for k, v in tokenized_context.items()}        # Check what you get

        n_context_tokens = tokenized_contexts["input_ids"].shape[-1]

        # encoder-decoder models have a hard-coded prompt with the context used for the encoder
        # In contrast, decoder-only models use the context as a prompt.
        if self.network.config.is_encoder_decoder:
            prompt_length = 1
            last = None
            torch.tensor([self.tokenizer.bos_token_id] * sampling_size).unsqueeze(-1).to(self.device)
            decoder_input_ids, generate_kwargs = self.network._prepare_decoder_input_ids_for_generation(sampling_size,
                    "decoder_input_ids",
                    generate_kwargs)
            generate_kwargs["decoder_input_ids"] = decoder_input_ids
        else:
            prompt_length = n_context_tokens
            last = None

        outputs = self.network.generate(**tokenized_contexts,
            output_scores=True, return_dict_in_generate=True,
            max_new_tokens=self.length,
            do_sample=True, generation_config=self.gen_config)

        all_logprobs = torch.stack(outputs.scores, dim=1).log_softmax(-1)  # [sampling_size, length, vocab]
        token_seq_logprobs = torch.gather(
                all_logprobs, 2, outputs.sequences[:, prompt_length:last][:, :, None]
            ).squeeze(-1) # [sampling_size, length]

        # we need to zero the (log-)scores of extra <eos>
        first_eos_indices = get_token_first_indices(
                outputs.sequences[:, prompt_length:last],  # starting at 1 to skip an eventual bos token
                self.tokenizer.eos_token_id
            )
        non_pad_tokens = torch.cat(
                (outputs.sequences[:, prompt_length:last][:, 0].unsqueeze(1),
                torch.where(
                        self.tokenizer.pad_token_id == outputs.sequences[:, prompt_length:last][:, 1:],
                        -1,
                        outputs.sequences[:, prompt_length:last][:, 1:])
                    ),
                dim=1
            )
        non_pad_log_scores = torch.where(-1 != non_pad_tokens, token_seq_logprobs, torch.tensor(0.).to(self.device))
        for i, ix in enumerate(first_eos_indices):
            non_pad_log_scores[i][0] = token_seq_logprobs[i][0]  # at least score one token
            if ix != -1: # if there an eos token
                non_pad_log_scores[i][ix] = token_seq_logprobs[i][ix]  # keep the first eos scores
                non_pad_log_scores[i][ix + 1:] = 0. # ignore everything after eos

        seq_logprobs = non_pad_log_scores.sum(dim=1) if sum else non_pad_log_scores

        output_tokens = outputs.sequences[:, prompt_length:] # [sampling_size, length]

        return (
                [TextSample(ots, self.tokenizer.decode(ots)) for ots in output_tokens],
                seq_logprobs
            )
