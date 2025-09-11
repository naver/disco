# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import torch.nn.functional as F
import copy
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LogitsProcessorList, GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)

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
            length=1024, device="auto", process_context_fn=None,
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
            flag to eventually (not) freeze the model's parameters
        length: int
            number of tokens in the samples
        device: string
            reference of the computing device
        process_context_fn: function
            a callback that processes the context before feeding it to the model
        config: kwarg
            parameters and values passed to transformers' ```generate(…)```
        """

        if isinstance(model, str):
            # assume also the tokenizer is a str
            self.tokenizer= AutoTokenizer.from_pretrained(tokenizer if tokenizer else model, padding_side="left")
            assert auto in [AutoModelForCausalLM, AutoModelForSeq2SeqLM], "only AutoModel, AutoModelForCausalLM and AutoModelForSeq2SeqLM are valid options."
            self._load_model(auto, model, device)
        else:
            self.tokenizer = tokenizer
            self.model = model
            self.device = model.device

        self.model.eval() # to make sure scoring is consistent
        if freeze:
            self.freeze(True)

        self.length = length

        self.process_context_fn = process_context_fn

        self.gen_config = GenerationConfig(**{
            "top_k": 0,
            "top_p": 1.0,
            "typical_p": 1.0,
            "temperature": 1.0,
            "num_beams": 1
        })
        self.gen_config.update(**config)

    def _load_model(self, auto, model, device):
        print(f"Loading model to {device}")
        t0 = time.time()
        self.model = auto.from_pretrained(model, device_map=device, trust_remote_code=True, use_safetensors=True)
        self.device = self.model.device
        print(f"Model loaded in {time.time() - t0:.0f}s.", )

    def validation(self):
        """"
        Return a shallow copy configured for validation sampling
        """
        val_self = copy.copy(self)
        val_self.gen_config.do_sample=False
        return val_self

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def freeze(self, frozen=True):
        """Freeze (or unfreeze) parameters for gradient computation.

        Parameters
        ----------
        frozen: boolean (True)
            state to transition to, default is to freeze
        """

        self.model.requires_grad_(not frozen)

    def string_to_textsample(self, text):
        """
        Convert a string to a TextSample namedtuple.

        Parameters
        ----------
        text: str
            The input text string to convert

        Returns
        -------
        TextSample
            A namedtuple containing token_ids and the original text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Tokenize the text
        token_ids = self.tokenizer.encode(text, return_tensors="pt").squeeze()

        return TextSample(token_ids=token_ids, text=text)

    def sample(self, context="", sampling_size=32, sum=True):
        """
        Samples sequences from the language model in the given context

        Parameters
        ----------
        context: str
            Contextual text for which to sample.
        sampling_size: int
            Number of sequences to sample.
        sum: bool
            Flag to return a token-level tensor of scores or sum them per sequence.

        Returns
        -------
        tuple of (list of TextSample(tokens, text), tensor of logprobs)
        """
        if self.model.config.is_encoder_decoder:
            assert context, "Context (encoder input) is mandatory for encoder-decoder models."
        elif not context:
            context = self.tokenizer.bos_token

        if self.process_context_fn:
            context = (self.process_context_fn)(context)

        tokenized_context = self.tokenizer(
            context,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        # Replicate the context for batch generation
        tokenized_contexts = {k: v.to(self.model.device).repeat(sampling_size, 1) for k, v in tokenized_context.items()}

        prompt_length = tokenized_contexts["input_ids"].shape[1]

        # Generate sequences and scores in one go
        outputs = self.model.generate(
            **tokenized_contexts,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=self.length,
            do_sample=True,
            generation_config=self.gen_config
        )

        generated_sequences = outputs.sequences[:, prompt_length:]

        logprobs_list = []
        for i, scores_at_step in enumerate(outputs.scores):
            generated_tokens_at_step = generated_sequences[:, i].unsqueeze(-1)
            logprobs_at_step = scores_at_step.log_softmax(dim=-1)
            token_logprob = torch.gather(logprobs_at_step, 1, generated_tokens_at_step)
            logprobs_list.append(token_logprob)
        token_seq_logprobs = torch.cat(logprobs_list, dim=1)

        # Create a mask to zero out logprobs for tokens after the first EOS token.
        # This entire block replaces the slow Python `for` loop.
        eos_token_id = self.tokenizer.eos_token_id
        is_eos = (generated_sequences == eos_token_id)

        # Find the index of the first EOS in each sequence
        # cumsum will be 0 before the first True, 1 at the first True, and >1 after.
        eos_mask = is_eos.cumsum(dim=1) <= 1

        # Also create a mask for padding tokens, if any
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            pad_mask = (generated_sequences != pad_token_id)
            final_mask = eos_mask & pad_mask
        else:
            final_mask = eos_mask

        # Apply the combined mask to zero out irrelevant scores
        final_logprobs = token_seq_logprobs.where(final_mask, 0.0)

        seq_logprobs = final_logprobs.sum(dim=1) if sum else final_logprobs

        # Decode all sequences
        decoded_texts = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        # Create the final list of samples
        samples = [TextSample(tokens, text) for tokens, text in zip(generated_sequences, decoded_texts)]

        return (samples, seq_logprobs)

    def sample_batch(self, contexts, sampling_size=32, sum=True, output_scores=True):
        """
        Generates samples for a batch of contexts.

        This method processes multiple input contexts simultaneously, generating
        a specified number of samples for each. It leverages batching to
        efficiently perform generation on a GPU.

        Parameters
        ----------
        contexts: list of str
            A list of contextual text strings for which to sample.
        sampling_size: int
            The number of sequences to sample for each context in the list.
        sum: bool
            If True, returns the sum of log probabilities for each sequence.
            If False, returns a tensor of token-level log probabilities.

        Returns
        -------
        tuple of (list of lists of TextSample, torch.Tensor)
            - A list of lists, where the outer list corresponds to the input
              contexts and each inner list contains `sampling_size` TextSample
              objects.
            - A tensor containing the log probabilities of the generated
              sequences, with shape `(num_contexts, sampling_size)` if `sum` is
              True, or `(num_contexts, sampling_size, sequence_length)` if
              `sum` is False.
        """
        if not isinstance(contexts, list) or not contexts:
            raise ValueError("contexts must be a non-empty list of strings.")

        if self.process_context_fn:
            contexts = [(self.process_context_fn)(context) for context in contexts]

        # Tokenize the entire batch of contexts with padding
        tokenized_contexts = self.tokenizer(
            contexts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        num_contexts = len(contexts)
        total_batch_size = num_contexts * sampling_size

        # Repeat the tokenized inputs to create a batch of size
        # (num_contexts * sampling_size) where each context is repeated
        # `sampling_size` times.
        batch = {
            "input_ids": tokenized_contexts["input_ids"].repeat_interleave(sampling_size, dim=0),
            "attention_mask": tokenized_contexts["attention_mask"].repeat_interleave(sampling_size, dim=0)
        }

        # Determine the length of the prompt to slice it off the output
        if self.model.config.is_encoder_decoder:
            prompt_length = 1
            last = None
        else:
            prompt_length = batch["input_ids"].shape[-1]
            last = None

        # Generate sequences for the entire batch
        outputs = self.model.generate(
            **batch,
            output_scores=output_scores,
            return_dict_in_generate=True,
            max_new_tokens=self.length,
            do_sample=True,
            generation_config=self.gen_config
        )

        # Process scores for the generated tokens
        generated_sequences = outputs.sequences[:, prompt_length:last]

        if output_scores:
            logprobs_list = []
            for i, scores_at_step in enumerate(outputs.scores):
                generated_tokens_at_step = generated_sequences[:, i].unsqueeze(-1)
                logprobs_at_step = scores_at_step.log_softmax(dim=-1)
                token_logprob = torch.gather(logprobs_at_step, 1, generated_tokens_at_step)
                logprobs_list.append(token_logprob)
            token_seq_logprobs = torch.cat(logprobs_list, dim=1)

            # Zero out log probabilities for padding tokens and any tokens
            # generated after the first end-of-sequence (EOS) token.
            first_eos_indices = get_token_first_indices(
                generated_sequences, self.tokenizer.eos_token_id
            )
            non_pad_tokens = torch.cat(
                (generated_sequences[:, 0].unsqueeze(1),
                torch.where(
                    self.tokenizer.pad_token_id == generated_sequences[:, 1:],
                    torch.tensor(-1, device=self.device),
                    generated_sequences[:, 1:])
                ),
                dim=1
            )
            non_pad_log_scores = torch.where(-1 != non_pad_tokens, token_seq_logprobs, torch.tensor(0., device=self.device))

            for i, ix in enumerate(first_eos_indices):
                non_pad_log_scores[i][0] = token_seq_logprobs[i][0]
                if ix != -1:
                    non_pad_log_scores[i][ix] = token_seq_logprobs[i][ix]
                    if ix + 1 < non_pad_log_scores.shape[1]:
                        non_pad_log_scores[i][ix + 1:] = 0.

            seq_logprobs_flat = non_pad_log_scores.sum(dim=1) if sum else non_pad_log_scores

            # Reshape the flat outputs back into a batched structure
            # corresponding to the input contexts.
            if sum:
                seq_logprobs = seq_logprobs_flat.view(num_contexts, sampling_size)
            else:
                seq_logprobs = seq_logprobs_flat.view(num_contexts, sampling_size, -1)

        # Extract the generated token IDs (excluding the prompt)
        output_tokens_flat = outputs.sequences[:, prompt_length:]

        # Decode all tokens and create TextSample objects
        all_samples_decoded = self.tokenizer.batch_decode(output_tokens_flat, skip_special_tokens=True)
        all_samples_flat = [
            TextSample(token_ids, text)
            for token_ids, text in zip(output_tokens_flat, all_samples_decoded)
        ]

        # Reshape the flat list of samples into a nested list (list of lists)
        all_samples_nested = [
            all_samples_flat[i * sampling_size : (i + 1) * sampling_size]
            for i in range(num_contexts)
        ]

        if output_scores:
            return all_samples_nested, seq_logprobs
        else:
            return all_samples_nested

    def log_score(self, samples, context="", grad=False, sum=True):
        """
        Computes log-probabilities for samples.
        """
        shapes = {s.token_ids.shape for s in samples}
        assert len(shapes) == 1, "All sequences of token_ids must have the same shape."

        if self.model.config.is_encoder_decoder:
            assert context, "Context is mandatory for encoder-decoder models."
        elif not context:
            context = self.tokenizer.bos_token

        if self.process_context_fn:
            context = (self.process_context_fn)(context)

        tokenized_context = self.tokenizer([context] * len(samples), return_tensors="pt", add_special_tokens=True, padding=True)
        tokenized_context = {k: v.to(self.device) for k, v in tokenized_context.items()}
        tokenized_samples = {"input_ids": torch.stack([s.token_ids for s in samples]).to(self.device)}
        tokenized_samples = self._discount_padding_tokens(tokenized_samples)
        _, _, forward_kwargs, labels, _, _ = self._get_forward_inputs(
            tokenized_context, tokenized_samples, samples
        )

        with torch.set_grad_enabled(grad):
            outputs = self.model(**forward_kwargs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        padding_mask = (shift_labels == -100)
        clean_shift_labels = shift_labels.clone()
        clean_shift_labels[padding_mask] = 0

        seq_logprobs = selective_log_softmax(shift_logits, clean_shift_labels)
        seq_logprobs.masked_fill_(padding_mask, 0.0)

        return seq_logprobs.sum(dim=1) if sum else seq_logprobs

    def _discount_padding_tokens(self, tokenized_samples):
        """
        Creates an attention mask that ignores padding, post-EOS tokens, and an optional forced BOS token.
        """
        input_ids = tokenized_samples["input_ids"]
        device = input_ids.device

        # 1. Create a base mask that ignores padding tokens. This is already vectorized.
        pad_mask = (input_ids != self.tokenizer.pad_token_id)

        # 2. Create a mask to ignore tokens after the first EOS token.
        # This replaces the main `for` loop.
        is_eos = (input_ids == self.tokenizer.eos_token_id)
        # The cumulative sum will be 0 before the first EOS, 1 at the first EOS, and >1 after.
        # By keeping everything <= 1, we keep all tokens up to and including the first EOS.
        eos_mask = is_eos.cumsum(dim=1) <= 1

        # 3. Combine the masks. A token is attended to if it's not a pad AND it's not after the first EOS.
        attention_mask = pad_mask & eos_mask

        # 4. Vectorized handling of a forced BOS token, if applicable.
        # This prevents scoring the model on predicting the BOS token it was given.
        forced_bos_token_id = getattr(self.model.config, "forced_bos_token_id", None)
        if forced_bos_token_id is not None:
            # Create a mask that is True only for the first token if it's a forced BOS
            is_forced_bos_at_start = (input_ids[:, 0] == forced_bos_token_id)
            # Set the attention mask at the first position to 0 where the condition is True
            attention_mask[:, 0] = torch.where(is_forced_bos_at_start, False, attention_mask[:, 0])

        tokenized_samples["attention_mask"] = attention_mask.to(torch.int) # Convert boolean to int
        return tokenized_samples

    def _get_forward_inputs(self, tokenized_context, tokenized_samples, samples):
        """
        Prepares model inputs and labels correctly and efficiently.
        This version includes an explicit check for the sample attention mask
        and clarifies label creation.
        """
        assert "attention_mask" in tokenized_samples, \
            "_get_forward_inputs requires 'attention_mask' in tokenized_samples. " \
            "Please call _discount_padding_tokens first."

        if self.model.config.is_encoder_decoder:
            encoder_inputs = tokenized_context
            decoder_inputs = tokenized_samples["input_ids"]

            forward_kwargs = {
                "input_ids": encoder_inputs["input_ids"],
                "attention_mask": encoder_inputs["attention_mask"],
                "decoder_input_ids": decoder_inputs,
            }

            # Create labels and use the pre-computed sample mask to ignore padding/EOS.
            labels = decoder_inputs.clone()
            labels[tokenized_samples["attention_mask"] == 0] = -100

            prompt_length, last = None, None
            encoder_input_ids = encoder_inputs["input_ids"]
            # The first returned value is ambiguous here, let's keep it for compatibility.
            # It's better to rely on forward_kwargs.
            input_ids = encoder_input_ids

        else: # Decoder-only models (e.g., GPT-2, Llama)
            context_ids = tokenized_context["input_ids"]
            sample_ids = tokenized_samples["input_ids"]
            input_ids = torch.cat((context_ids, sample_ids), dim=1)

            context_mask = tokenized_context["attention_mask"]
            sample_mask = tokenized_samples["attention_mask"]
            attention_mask = torch.cat((context_mask, sample_mask), dim=1)

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            labels = input_ids.clone()
            prompt_length = context_ids.shape[1]

            # 1. Ignore the context part of the labels.
            labels[:, :prompt_length] = -100

            # 2. Ignore the padded/post-EOS part of the samples in the labels.
            # We create a boolean mask for the sample part of the labels.
            sample_part_in_labels_mask = torch.zeros_like(labels, dtype=torch.bool)
            sample_part_in_labels_mask[:, prompt_length:] = True

            # The sample_mask tells us which tokens in the sample are invalid.
            # We combine these conditions: a label is ignored if it's in the sample part
            # AND its corresponding sample_mask is 0.
            labels[(sample_part_in_labels_mask) & (attention_mask == 0)] = -100

            last, encoder_input_ids = None, None

        return input_ids, encoder_input_ids, forward_kwargs, labels, prompt_length, last

    def _process_and_warp_logits(self, all_logits, input_ids, encoder_input_ids):
        """
        Applies logits processors to the full logits tensor in a highly optimized way.

        This version separates stateful and stateless processors. Stateless ones are applied
        in a single vectorized operation, while the slow sequential loop is only used for
        the few stateful ones that require it.
        """
        # 1. Get the full list of processors as before
        generation_config = copy.deepcopy(self.gen_config)
        # The _prepare_special_tokens call is often unnecessary here, but we keep for compatibility
        self.model._prepare_special_tokens(generation_config, False, self.model.device)

        full_processor_list = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList()
        )

        if not full_processor_list:
            return all_logits

        # 2. Separate processors into stateless and stateful lists
        stateless_processors = LogitsProcessorList()
        stateful_processors = LogitsProcessorList()

        # Define which processor classes are stateful (require token history)
        STATEFUL_PROCESSOR_CLASSES = (
            RepetitionPenaltyLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
        )

        for processor in full_processor_list:
            if isinstance(processor, STATEFUL_PROCESSOR_CLASSES):
                stateful_processors.append(processor)
            else:
                stateless_processors.append(processor)

        # 3. Apply all STATELESS processors in a single, vectorized call.
        # Most processors (Temperature, TopK, TopP) are stateless and can operate
        # on the entire [batch, seq_len, vocab_size] tensor at once.
        if stateless_processors:
            # Note: The `input_ids` argument is ignored by stateless processors.
            all_logits = stateless_processors(None, all_logits)

        # 4. Apply STATEFUL processors sequentially, ONLY if necessary.
        # This is the original slow loop, but now it runs with a much smaller
        # list of processors, or not at all if none are stateful.
        if stateful_processors:
            processed_logits = []
            for i in range(all_logits.shape[1]):
                # The input history is `input_ids[:, :i+1]`, and current logits are `all_logits[:, i, :]`
                current_step_logits = all_logits[:, i, :]
                history = input_ids[:, :i+1]
                processed_step_logits = stateful_processors(history, current_step_logits)
                processed_logits.append(processed_step_logits.unsqueeze(1))

            all_logits = torch.cat(processed_logits, dim=1)

        return all_logits

    def log_score_batch(self, samples, contexts, grad=False, sum=True):
        """
        Computes log-probabilities for a batch of samples, each associated with a
        corresponding context from a list.
        """
        assert len(contexts) > 0
        assert len(samples) == len(contexts)
        assert len(samples[0]) > 0, "Each context must have at least one corresponding sample."

        if self.process_context_fn:
            contexts = [(self.process_context_fn)(context) for context in contexts]

        num_contexts = len(contexts)
        n_samples_per_context = len(samples[0])

        tokenized_contexts_unique = self.tokenizer(
            contexts, return_tensors="pt", add_special_tokens=True, padding=True, padding_side="left"
        )
        samples_flat = [s for sublist in samples for s in sublist]
        tokenized_context = {
            k: v.repeat_interleave(n_samples_per_context, dim=0).to(self.device)
            for k, v in tokenized_contexts_unique.items()
        }
        tokenized_samples = {"input_ids": torch.stack([s.token_ids for s in samples_flat]).to(self.device)}
        tokenized_samples = self._discount_padding_tokens(tokenized_samples)
        _, _, forward_kwargs, labels, _, _ = self._get_forward_inputs(
            tokenized_context, tokenized_samples, samples_flat
        )

        with torch.set_grad_enabled(grad):
            outputs = self.model(**forward_kwargs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        padding_mask = (shift_labels == -100)
        clean_shift_labels = shift_labels.clone()
        clean_shift_labels[padding_mask] = 0

        seq_logprobs_flat = selective_log_softmax(shift_logits, clean_shift_labels)
        seq_logprobs_flat.masked_fill_(padding_mask, 0.0)

        if sum:
            summed_logprobs = seq_logprobs_flat.sum(dim=1)
            return summed_logprobs.view(num_contexts, n_samples_per_context)
        else:
            return seq_logprobs_flat.view(num_contexts, n_samples_per_context, -1)

    def report_samples_stats(self, samples, contexts, observable):
        """
        Reports statistics about the samples to the observable
        """
        pad_token_id = self.tokenizer.pad_token_id

        if self.process_context_fn:
            contexts = [(self.process_context_fn)(context) for context in contexts]

        tokenized_contexts = self.tokenizer(contexts)
        for context_ids in tokenized_contexts:
            observable.dispatch("context_length", len(context_ids))

        for sample in samples:
            length_with_padding = len(sample.token_ids)
            length_without_padding = sum(1 for token_id in sample.token_ids if token_id != pad_token_id)
            observable.dispatch("sample_length_with_padding", length_with_padding)
            observable.dispatch("sample_length_without_padding", length_without_padding)

# From https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
def selective_log_softmax(logits, index):
    """Compute log softmax probabilities for selected tokens.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.
    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])  # loop to reduce peak mem consumption
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        token_logprobs = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        token_logprobs = []
        for logits_row, index_row in zip(logits, index):  # loop to reduce peak mem consumption
            logprobs_row = logits_row.log_softmax(dim=-1)
            token_logprobs_row = torch.gather(logprobs_row, dim=-1, index=index_row.unsqueeze(-1)).squeeze(-1)
            token_logprobs.append(token_logprobs_row)
        token_logprobs = torch.stack(token_logprobs)
    return token_logprobs
