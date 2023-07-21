# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import unittest
import random, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from disco.distributions import LMDistribution
from disco.distributions.lm_distribution import TextSample

prefix = "It was a cold and stormy night"

class LMDistributionTest(unittest.TestCase):

    def setUp(self):
        test_models = [("gpt2", AutoModelForCausalLM),
                ("facebook/bart-base", AutoModelForSeq2SeqLM)]
        self.test_distributions = {model: LMDistribution(model, auto=auto)
                for (model, auto) in test_models}

    def test_instantiate_a_default_distribution(self):
        distribution = LMDistribution()
        self.assertTrue(hasattr(distribution, "length"),
            "the distribution should have a length attribute.")
        self.assertEqual(40, distribution.length,
            "the default length should be 40.")
        self.assertTrue(hasattr(distribution, "params"),
            "the distribution should have a params attribute.")
        self.assertTrue(hasattr(distribution, "scorable"),
            "the distribution should have a scorable attribute.")
        self.assertEqual(True, distribution.scorable,
            "the distribution should be scorable by default.")

    def test_sample_a_continuation_from_a_default_distribution(self):
        distribution = LMDistribution()
        samples, log_scores = distribution.sample(context=prefix)
        self.assertTrue(isinstance(samples, list), "the samples should be returned as a list.")
        from disco.distributions.lm_distribution import TextSample
        for sample in samples:
            self.assertTrue(isinstance(sample, TextSample), "each text should be a textSample.")
        self.assertTrue(isinstance(log_scores, torch.Tensor), "the log-scores should be returned as a tensor.")
        for log_score in log_scores:
            self.assertTrue(isinstance(log_score.item(), float), "each log-score should be a float.")

    def test_sample_multiple_continuations_for_a_prefix(self):
        distribution = LMDistribution()
        sampling_size = 8
        samples, log_scores = distribution.sample(sampling_size=sampling_size)
        self.assertEqual(sampling_size, len(samples),
            "the number of returned samples should be {}.".format(sampling_size))
        self.assertEqual(sampling_size, len(log_scores),
            "the number of returned log_scores should be {}.".format(sampling_size))

    def test_sample_a_continuation_with_temperature_close_to_zero(self):
        distribution = LMDistribution(temperature=0.001)
        samples, log_scores = distribution.sample(sampling_size=2, context=prefix)
        self.assertEqual(samples[0].text, samples[1].text,
            "samples should not vary when temperature is (almost) zero.")
        self.assertEqual(log_scores[0], log_scores[1],
            "log_scores should not vary when temperature is (almost) zero.")

    def test_score_sequences(self):
        prompt = "It was a cold and stormy night"
        distribution = LMDistribution()
        texts = [
                " and the streets were empty and quiet.",
                "; the rain fell in torrents"
            ]
        tokenized_texts = distribution.tokenizer(texts, return_tensors="pt", add_special_tokens=True, padding=True)
        from disco.distributions.lm_distribution import TextSample
        samples = [TextSample(s, t) for (s, t) in zip(tokenized_texts["input_ids"], texts)]
        log_scores = distribution.log_score(samples, context=prefix)
        self.assertTrue(isinstance(log_scores, torch.Tensor), "the log-scores should be returned as a tensor.")
        self.assertEqual(len(samples), len(log_scores), "there should be as many log-scores as there are sequences.")
        for log_score in log_scores:
            self.assertTrue(isinstance(log_score.item(), float), "each log-score should be a float.")
            self.assertLess(log_score, 0.0, "each log-score should be negative.")

    def test_score_sampled_sequences(self):
        for model, distribution in self.test_distributions.items():
            prompt = "It was a cold and stormy night"
            distribution = LMDistribution()
            samples, log_scores = distribution.sample(context=prefix)
            samples_log_scores_again = distribution.log_score(samples, context=prefix)
            self.assertLess((log_scores - samples_log_scores_again).abs().max(),
                    1e-3, "log-scores given at sampling time and the scoring of "
                    "the same samples should match")

    def test_score_with_a_non_default_top_k_specified(self):
        distribution = LMDistribution(top_k=20)
        samples, _ = distribution.sample(context=prefix)
        self.assertRaises(AssertionError, distribution.log_score, samples, context=prefix)

    def test_score_with_a_non_default_top_p_specified(self):
        distribution = LMDistribution(top_p=0.92)
        samples, _ = distribution.sample(context=prefix)
        self.assertRaises(AssertionError, distribution.log_score, samples, context=prefix)

    def test_score_with_a_non_default_typical_p_specified(self):
        distribution = LMDistribution(typical_p=0.9)
        samples, _ = distribution.sample(context=prefix)
        self.assertRaises(AssertionError, distribution.log_score, samples, context=prefix)

    def test_score_with_a_non_default_temperature_specified(self):
        distribution = LMDistribution(temperature=0.7)
        samples, _ = distribution.sample(context=prefix)
        self.assertRaises(AssertionError, distribution.log_score, samples, context=prefix)

    def test_score_sequences_different_length(self):
        distribution = LMDistribution()
        texts = [
                " and the streets were empty and quiet.",
                " in Brest."
            ]

        from disco.distributions.lm_distribution import TextSample
        samples = [
                TextSample(distribution.tokenizer(t, return_tensors="pt", add_special_tokens=True)["input_ids"], t)\
                    for t in texts
            ]
        self.assertRaises(AssertionError, distribution.log_score, samples, context=prefix)

    def test_ignore_padding_at_score(self):
        for model, distribution in self.test_distributions.items():
            text = "The streets were empty and quiet."
            context = "This is some context." if distribution.network.config.is_encoder_decoder else ""

            from disco.distributions.lm_distribution import TextSample
            sample = TextSample(distribution.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0], text)
            eos_token_id = distribution.tokenizer.eos_token_id
            pad_token_id = distribution.tokenizer.pad_token_id

            eos_token = distribution.tokenizer.eos_token
            pad_token = distribution.tokenizer.pad_token
            completed_sample = TextSample(
                    token_ids=torch.cat((sample.token_ids, torch.tensor([eos_token_id])), dim=0),
                    text=text + eos_token)

            padded_sample = TextSample(
                    token_ids=torch.cat((sample.token_ids, torch.tensor([eos_token_id, pad_token_id])), dim=0),
                    text=text + pad_token + pad_token)

            log_scores = distribution.log_score([sample], context=context).item()
            completed_log_scores = distribution.log_score([completed_sample], context=context).item()
            padded_log_scores = distribution.log_score([padded_sample], context=context).item()

            self.assertNotEqual(0, log_scores)
            if completed_log_scores != float("-inf") and completed_log_scores != float("-inf"):
                self.assertNotAlmostEqual(log_scores, completed_log_scores, 4,
                    f"The {model} scores must be different with and without and eos token.")
            self.assertAlmostEqual(completed_log_scores, padded_log_scores, 4,
                    f"The {model} scores must not be different with and without a pad token after an eos token.")

    def test_ignore_padding_at_sample(self):
        distribution = LMDistribution()
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        samples, log_scores = distribution.sample()
        padded_sequence_id = 16 # found manually
        padded_sequence = samples[padded_sequence_id]
        padded_sequence_log_score = log_scores[padded_sequence_id]
        pad_token_id = distribution.tokenizer.pad_token_id
        self.assertTrue((padded_sequence.token_ids[1:] == pad_token_id).all())
        padded_sequence_relog_score = distribution.log_score([padded_sequence])
        self.assertLess(torch.abs(padded_sequence_log_score - padded_sequence_relog_score), 1e-2)

    def test_empty_sequence_score(self):
        for model, distribution in self.test_distributions.items():
            pad_token_id = distribution.tokenizer.pad_token_id
            empty_sequence1_tok = [pad_token_id]*1
            empty_sequence1_str = distribution.tokenizer.decode(empty_sequence1_tok)
            empty_sequence1 = TextSample(token_ids=torch.tensor(empty_sequence1_tok),
                    text=empty_sequence1_str)
            empty_sequence2_tok = [pad_token_id]*20
            empty_sequence2_str = distribution.tokenizer.decode(empty_sequence2_tok)
            empty_sequence2 = TextSample(token_ids=torch.tensor(empty_sequence2_tok),
                    text=empty_sequence2_str)
            if distribution.network.config.is_encoder_decoder:
                log_score1 = distribution.log_score([empty_sequence1], context='yada yada')
                log_score2 = distribution.log_score([empty_sequence2], context='yada yada')
            else:
                log_score1 = distribution.log_score([empty_sequence1])
                log_score2 = distribution.log_score([empty_sequence2])
            self.assertAlmostEqual(log_score1.item(), log_score2.item(), 4,
                    f"The {model} score of all pad tokens should correspond to a single one.")

    def test_scoring_consistent_with_loss(self):
        for model, distribution in self.test_distributions.items():
            text = "The streets were empty and quiet."
            context = "This is an example." if distribution.network.config.is_encoder_decoder else ""

            from disco.distributions.lm_distribution import TextSample
            sample = TextSample(distribution.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0], text)
            log_score = distribution.log_score([sample], context=context)
            if distribution.network.config.is_encoder_decoder:
                ctxt = distribution.tokenizer(context, return_tensors="pt", add_special_tokens=True)
                loss = distribution.network(input_ids=ctxt.input_ids,
                        labels=sample.token_ids.unsqueeze(0)).loss
            else:
                loss = distribution.network(sample.token_ids, labels=sample.token_ids).loss
            self.assertLess(torch.abs(-log_score / len(sample.token_ids) -  loss), 1e-2,
                    f'The {model} score is not consistent with the loss reported by the forward function.')

    def test_sample_empty_sequence(self):
        distribution = LMDistribution("gpt2")
        pad_token_id = distribution.tokenizer.pad_token_id
        epsilon = 0.05
        # increase the likelihood of sampling pad_token_id
        distribution.network.lm_head.weight.data[pad_token_id, :] += epsilon
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        samples, log_scores = distribution.sample()
        self.assertTrue(any((s.token_ids == pad_token_id).all() for s in samples))
        new_log_scores = distribution.log_score(samples)
        self.assertLess(torch.abs(log_scores - new_log_scores).max(), 1e-4)

    def test_freeze_parameters_by_default(self):
        distribution = LMDistribution()
        self.assertTrue(all([not p.requires_grad for p in distribution.network.parameters()]))

    def test_unfreeze_parameters_on_demand(self):
        distribution = LMDistribution()
        distribution.freeze(False)
        self.assertTrue(all([p.requires_grad for p in distribution.network.parameters()]))

    def test_unfreeze_all_parameters_with_parameter(self):
        distribution = LMDistribution(freeze=False)
        self.assertTrue(all([p.requires_grad for p in distribution.network.parameters()]))

    def test_freeze_all_parameters_on_demand(self):
        distribution = LMDistribution(freeze=False)
        distribution.freeze()
        self.assertTrue(all([not p.requires_grad for p in distribution.network.parameters()]))

    def test_clone(self):
        distribution = LMDistribution()
        distribution2 = distribution.clone()
        self.assertEqual(distribution.network.config, distribution2.network.config)
        self.assertNotEqual(distribution.network, distribution2.network)

    def test_bart_consistency(self):
        model = LMDistribution(
                model="facebook/bart-large-cnn",
                auto=AutoModelForSeq2SeqLM,
                length=128)

        context="""Self-trained autonomous agents developed using machine learning are showing great promise in a variety of control settings,
        perhaps most remarkably in applications involving autonomous vehicles. The main challenge associated with self-learned agents in the form
        of deep neural networks, is their black-box nature: it is impossible for humans to interpret deep neural networks. Therefore, humans cannot
        directly interpret the actions of deep neural network based agents, or foresee their robustness in different scenarios. In this work,
        we demonstrate a method for probing which concepts self-learning agents internalise in the course of their training. For demonstration,
        we use a chess playing agent in a fast and light environment developed specifically to be suitable for research groups without access to
        enormous computational resources or machine learning models."""
        torch.manual_seed(1)
        samples, log_scores = model.sample(context=context, sampling_size=1, sum=False)
        ls_log_scores = model.log_score(samples, context=context, sum=False)
        sample_idx = 0
        self.assertAlmostEqual(log_scores.sum().item(), ls_log_scores.sum().item(), 4)

if __name__ == '__main__':
    unittest.main()
