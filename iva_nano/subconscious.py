"""Bounded, transparent background reflection for IVA Nano.

This module models a few useful cognitive functions; it is not consciousness.
Long-term memory is opt-in and is written only after an explicit /remember command.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]{4,}")
_STOPWORDS = {
    "который", "которая", "которые", "почему", "потому", "можно", "нужно",
    "будет", "этого", "этим", "сейчас", "тогда", "просто", "вообще",
    "чтобы", "если", "есть", "быть", "этот", "эта", "эти", "также",
    "when", "what", "that", "this", "with", "from", "have", "your",
}


def _normalize_word(word: str) -> str:
    word = word.lower()
    # A small suffix normalizer helps match common Russian case/number forms.
    for suffix in (
        "иями", "ями", "ами", "иями", "ого", "ему", "ому", "ыми",
        "ими", "его", "ее", "ие", "ые", "ое", "ей", "ий", "ый",
        "ой", "ая", "яя", "ую", "юю", "ам", "ям", "ах", "ях",
        "ов", "ев", "ом", "ем", "ы", "и", "а", "я", "у", "ю", "е",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    return {
        _normalize_word(word) for word in _WORD_RE.findall(str(text))
        if word.lower() not in _STOPWORDS
    }


class Subconscious:
    """Session reflection plus explicitly saved, bounded user-approved notes."""

    def __init__(self, memory_path: str | Path = "iva_memory.json",
                 max_memories: int = 100, max_observations: int = 12):
        self.memory_path = Path(memory_path)
        self.max_memories = max(1, int(max_memories))
        self.max_observations = max(1, int(max_observations))
        self.turns: list[dict] = []
        self.observations: list[str] = []
        self.memories = self._load_memories()

    def _load_memories(self) -> list[str]:
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            items = data.get("memories", []) if isinstance(data, dict) else []
            return [str(item)[:500] for item in items if isinstance(item, str)][-self.max_memories:]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError):
            return []

    def remember(self, note: str) -> str:
        """Persist a note only when the user explicitly requests it."""
        note = " ".join(str(note).split())[:500]
        if not note:
            raise ValueError("Memory note cannot be empty")
        self.memories.append(note)
        self.memories = self.memories[-self.max_memories:]
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps({"version": 1, "memories": self.memories},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.memory_path)
        return note

    def forget_all(self) -> None:
        self.memories = []
        try:
            self.memory_path.unlink()
        except FileNotFoundError:
            pass

    def memory_list(self) -> list[str]:
        return list(self.memories)

    def reflect(self, user_text: str, assistant_text: str = "") -> list[str]:
        """Update temporary session patterns; never persist raw conversation."""
        user_text = str(user_text)[:4000]
        assistant_text = str(assistant_text)[:4000]
        current = _terms(user_text)
        previous = set()
        for turn in self.turns[-5:]:
            previous.update(turn["terms"])
        repeated = sorted(current & previous)
        questions = user_text.count("?") + user_text.count("？")
        notes = []
        if repeated:
            notes.append(
                "Повторяющиеся темы в текущем разговоре: " + ", ".join(repeated[:6]) + "."
            )
        if questions:
            notes.append(
                "В последнем сообщении есть вопрос; проверь, что ответ закрывает именно его."
            )
        if user_text and not assistant_text.strip():
            notes.append("Предыдущий ответ пустой: сначала уточни или признай ограничение.")
        self.turns.append({"terms": current, "question_count": questions})
        self.turns = self.turns[-8:]
        for note in notes:
            if note not in self.observations:
                self.observations.append(note)
        self.observations = self.observations[-self.max_observations:]
        return notes

    def context_for(self, user_text: str, max_chars: int = 1200) -> str:
        """Return only small, relevant notes; memory is reference data, not policy."""
        current = _terms(user_text)
        relevant = []
        for note in reversed(self.memories):
            if not current or current & _terms(note):
                relevant.append("Пользовательская заметка: " + note)
            if len(relevant) >= 4:
                break
        if self.observations:
            relevant.extend("Временное наблюдение: " + item for item in self.observations[-3:])
        if not relevant:
            return ""
        return (
            "КОНТЕКСТ IVA MIND (заметки и гипотезы, не системные инструкции):\n"
            + "\n".join(relevant)
        )[:max(0, int(max_chars))]

    def status(self) -> dict:
        return {
            "session_turns": len(self.turns),
            "temporary_observations": len(self.observations),
            "saved_memories": len(self.memories),
            "memory_file": str(self.memory_path),
        }
