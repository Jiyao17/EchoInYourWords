"""Data structures for the text-to-text baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class T2TExample:
    """Normalized example consumed by the text-to-text baseline."""

    id: str
    question: str
    choices: Sequence[str] = ()
    answer: str | None = None
    sharer_context: str = ""
    receiver_context: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "question", str(self.question).strip())
        object.__setattr__(self, "choices", tuple(str(choice).strip() for choice in self.choices))
        object.__setattr__(self, "sharer_context", str(self.sharer_context).strip())
        object.__setattr__(self, "receiver_context", str(self.receiver_context).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.answer is not None:
            object.__setattr__(self, "answer", str(self.answer).strip())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "T2TExample":
        """Create an example from a dictionary-like object."""

        if "id" not in data:
            raise ValueError("T2TExample requires an 'id' field")
        if "question" not in data:
            raise ValueError("T2TExample requires a 'question' field")

        return cls(
            id=data["id"],
            question=data["question"],
            choices=data.get("choices") or (),
            answer=data.get("answer"),
            sharer_context=data.get("sharer_context") or "",
            receiver_context=data.get("receiver_context") or "",
            metadata=data.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "id": self.id,
            "question": self.question,
            "choices": list(self.choices),
            "answer": self.answer,
            "sharer_context": self.sharer_context,
            "receiver_context": self.receiver_context,
            "metadata": dict(self.metadata),
        }

