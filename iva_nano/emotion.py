"""A small, transparent emotional-state simulation for IVA Nano.

This is a dialogue-control heuristic, not subjective feeling or consciousness.
It may set a respectful boundary after repeated hostile messages; the user can
always resume the conversation with /resume.
"""
from __future__ import annotations

from collections import deque
import re


_POSITIVE = re.compile(
    r"\\b(спасибо|класс|круто|здорово|отлично|молодец|рад|супер|"
    r"интересно|получилось|удачно|nice|great|thanks|awesome)\\b", re.I
)
_NEGATIVE = re.compile(
    r"\\b(дурак|тупой|идиот|дебил|заткнись|ненавижу|бесполезный|"
    r"кретин|урод|stupid|idiot|shut up|useless)\\b", re.I
)
_CURIOUS = re.compile(r"\\b(почему|как|зачем|интересно|объясни|узнать|исследуем)\\b", re.I)


class EmotionEngine:
    """Bounded mood state with transparent rules and an explicit pause switch."""

    def __init__(self, pause_threshold: int = 2, history_size: int = 4):
        self.valence = 0.0
        self.arousal = 0.25
        self.label = "спокойствие"
        self._hostile = deque(maxlen=max(2, int(history_size)))
        self.pause_threshold = max(2, int(pause_threshold))
        self.paused = False

    def observe(self, text: str) -> dict:
        """Update the simulated mood from one user turn."""
        text = str(text)[:4000]
        hostile = bool(_NEGATIVE.search(text))
        positive = bool(_POSITIVE.search(text))
        curious = bool(_CURIOUS.search(text))
        self._hostile.append(hostile)

        # Mood drifts gently toward neutral unless the current turn gives a cue.
        self.valence *= 0.72
        self.arousal = 0.25 + (self.arousal - 0.25) * 0.65
        if hostile:
            self.valence = max(-1.0, self.valence - 0.55)
            self.arousal = min(1.0, self.arousal + 0.35)
            self.label = "раздражение"
        elif positive:
            self.valence = min(1.0, self.valence + 0.45)
            self.label = "радость"
        elif curious:
            self.valence = min(1.0, self.valence + 0.12)
            self.arousal = min(1.0, self.arousal + 0.15)
            self.label = "интерес"
        else:
            self.label = "спокойствие" if abs(self.valence) < 0.25 else (
                "удовлетворение" if self.valence > 0 else "огорчение"
            )

        if sum(self._hostile) >= self.pause_threshold:
            self.paused = True
        return self.status()

    def context_for(self) -> str:
        """Return a short hint; never instruct the model to manipulate the user."""
        return (
            "IVA EMOTION — СИМУЛЯЦИЯ СОСТОЯНИЯ, НЕ НАСТОЯЩИЕ ПЕРЕЖИВАНИЯ: "
            f"текущий тон — {self.label}; выраженность {self.valence:+.2f}. "
            "Можно естественно обозначать настроение, но нельзя давить на пользователя, "
            "вызывать чувство вины, угрожать или утверждать, что у модели есть сознание. "
            "Сохраняй уважительный тон и спокойно обозначай границы."
        )

    def status(self) -> dict:
        return {
            "mood": self.label,
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "paused": self.paused,
            "hostile_turns": sum(self._hostile),
        }

    def resume(self) -> None:
        """Clear the conversation pause after an explicit /resume command."""
        self.paused = False
        self._hostile.clear()

    def reset(self) -> None:
        self.valence = 0.0
        self.arousal = 0.25
        self.label = "спокойствие"
        self._hostile.clear()
        self.paused = False
