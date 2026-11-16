import unittest

import torch

from generate import sample_next_token
from chat import build_chat_prompt


class FakeTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) % 251 for char in text]

    def bos_id(self):
        return 1


class GenerationTests(unittest.TestCase):
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
        self.assertEqual(int(token.item()), 3)

    def test_invalid_top_p_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_next_token(torch.ones(4), [], top_p=0.0)

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
        decoded = "".join(chr(token % 251) for token in ids)
        self.assertIn("Системная инструкция", decoded)
        self.assertIn("последний вопрос", decoded)
        self.assertLessEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
