import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch

from generate import generate, sample_next_token
from chat import build_chat_prompt, print_iva_answer
from iva_nano.data import encode_training_texts


class FakeTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) for char in text]

    def bos_id(self):
        return 1

    def eos_id(self):
        return 3


class EosModel(torch.nn.Module):
    context_length = 32

    def forward(self, input_ids):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 8))
        logits[:, :, 3] = 10.0
        return logits, None


class GenerationTests(unittest.TestCase):
    def test_generation_stops_on_tokenizer_eos_and_reports_reason(self):
        stats = {}
        answer = generate(
            EosModel(), FakeTokenizer(), "prompt", "cpu",
            max_new_tokens=12, temperature=0.7, top_k=1, top_p=1.0,
            stats=stats,
        )
        self.assertEqual(answer, "")
        self.assertEqual(stats["eos_id"], 3)
        self.assertEqual(stats["generated_tokens"], 0)
        self.assertEqual(stats["stop_reason"], "eos")

    def test_generation_reports_length_limit_when_eos_not_emitted(self):
        class NonEosModel(torch.nn.Module):
            context_length = 32

            def forward(self, input_ids):
                logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 128))
                logits[:, :, 65] = 10.0
                return logits, None

        stats = {}
        generate(
            NonEosModel(), FakeTokenizer(), "prompt", "cpu",
            max_new_tokens=2, temperature=0.7, top_k=1, top_p=1.0,
            no_repeat_ngram_size=0, stats=stats,
        )
        self.assertEqual(stats["generated_tokens"], 2)
        self.assertEqual(stats["stop_reason"], "max_new_tokens")

    def test_iva_answer_is_printed_as_a_standalone_final_block(self):
        output = StringIO()
        with redirect_stdout(output):
            print_iva_answer("Это ответ самой IVA.")
        rendered = output.getvalue()
        self.assertIn("Ответ IVA (сгенерировано самой моделью):", rendered)
        self.assertTrue(rendered.rstrip().endswith("Это ответ самой IVA."))

    def test_top_k_one_selects_highest_logit(self):
        logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
        token = sample_next_token(
            logits, [], temperature=1.0, top_k=1, top_p=1.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        self.assertEqual(int(token.item()), 2)

    def test_repeated_trigram_is_blocked(self):
        logits = torch.tensor([0.0, 1.0, 9.0, 8.0, 7.0])
        token = sample_next_token(
            logits, [1, 2, 3, 1, 2], temperature=1.0, top_k=1,
            top_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=3,
        )
        # The existing suffix is [1, 2], so token 3 would repeat [1, 2, 3].
        self.assertNotEqual(int(token.item()), 3)
        self.assertEqual(int(token.item()), 2)

    def test_invalid_top_p_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_next_token(torch.ones(4), [], top_p=0.0)

    def test_sft_masks_user_prompt_and_trains_answer_only(self):
        tokenizer = FakeTokenizer()
        text = "### Пользователь:\nПривет\n### IVA:\nЗдравствуйте"
        token_ids, labels = encode_training_texts([text], tokenizer, assistant_only=True)
        prompt_length = len(tokenizer.encode(
            "### Пользователь:\nПривет\n### IVA:\n", out_type=int
        ))
        self.assertEqual(labels[:prompt_length], [-100] * prompt_length)
        self.assertEqual(labels[prompt_length:-1], token_ids[prompt_length:-1])
        self.assertEqual(labels[-1], tokenizer.eos_id())

    def test_chat_prompt_keeps_system_and_latest_user_turn(self):
        tokenizer = FakeTokenizer()
        messages = [
            ("user", "старое сообщение " * 20),
            ("assistant", "старый ответ " * 20),
            ("user", "последний вопрос"),
        ]
        ids = build_chat_prompt(
            tokenizer, messages, "Системная инструкция", context_length=120,
            reserve_tokens=20,
        )
        decoded = "".join(chr(token) for token in ids)
        self.assertIn("Системная инструкция", decoded)
        self.assertIn("последний вопрос", decoded)
        self.assertLessEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
