import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src import models
from src.utils.formatting import parse_datetime_iso


def _strip_microseconds_and_timezone(timestamp: datetime) -> datetime:
    """
    Remove microseconds and timezone info from a datetime for stable string formatting.
    """
    return timestamp.replace(microsecond=0, tzinfo=None)


def flatten_message_ids(
    message_ids: list[int] | list[list[int]] | list[tuple[int, int]],
) -> list[int]:
    """
    Flatten message_ids that may be in old tuple format or nested list format.

    This handles backwards compatibility with the old schema where message_ids
    was list[tuple[int, int]] representing ranges, and the new schema where
    it's list[int] representing individual message IDs.

    Args:
        message_ids: Either a flat list of ints, nested list, or list of tuples

    Returns:
        A flat list of unique message IDs, sorted

    Examples:
        [1, 2, 3] -> [1, 2, 3]
        [[1, 2], [3, 4]] -> [1, 2, 3, 4]
        [(105, 105)] -> [105]
        [[105, 105]] -> [105]
    """
    result: list[int] = []
    for item in message_ids:
        if isinstance(item, (list | tuple)):
            result.extend(item)
        else:
            result.append(item)
    return sorted(set(result))


class ObservationMetadata(BaseModel):
    id: str = Field(default="", description="Document ID for this observation")
    created_at: datetime
    message_ids: list[int]
    session_name: str | None = None


class ExplicitObservationBase(BaseModel):
    content: str = Field(description="The explicit observation")


class DeductiveObservationBase(BaseModel):
    source_ids: list[str] = Field(
        description="Document IDs of premise observations for tree traversal",
        default_factory=list,
    )
    premises: list[str] = Field(
        description="Human-readable premise text for display",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class InductiveObservationBase(BaseModel):
    """Base model for inductive observations - patterns, generalizations, and personality insights."""

    source_ids: list[str] = Field(
        description="Document IDs of source observations for tree traversal",
        default_factory=list,
    )
    sources: list[str] = Field(
        description="Human-readable source text for display",
        default_factory=list,
    )
    pattern_type: str = Field(
        description="Type of pattern: 'preference', 'behavior', 'personality', 'tendency', 'correlation'",
        default="pattern",
    )
    conclusion: str = Field(description="The inductive generalization or pattern")
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', 'low'",
        default="medium",
    )


class ContradictionObservationBase(BaseModel):
    """Base model for contradiction observations - when user has made conflicting statements."""

    source_ids: list[str] = Field(
        description="Document IDs of the contradicting observations",
        default_factory=list,
    )
    sources: list[str] = Field(
        description="Human-readable text of the contradicting statements",
        default_factory=list,
    )
    content: str = Field(description="Description of the contradiction")


class PromptRepresentation(BaseModel):
    """The representation format used when getting structured output from an LLM."""

    explicit: list[ExplicitObservationBase] = Field(
        description="Facts directly stated by the user, expressed as self-contained paraphrases.",
        default_factory=list,
    )

    @field_validator("explicit", mode="before")
    @classmethod
    def convert_none_to_empty_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, list):
            normalized: list[dict[str, str] | ExplicitObservationBase] = []
            for item in v:
                if isinstance(item, str):
                    content = item.strip()
                    if content:
                        normalized.append({"content": content})
                    continue
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    content = item["content"].strip()
                    if content:
                        normalized.append({"content": content})
                    continue
                normalized.append(item)
            return normalized
        return v

    @classmethod
    def from_plain_text(cls, text: str) -> "PromptRepresentation":
        observations: list[ExplicitObservationBase] = []
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = re.sub(r"^[-*•\d\.)\s]+", "", candidate).strip()
            if not candidate or candidate in {"{", "}", "[", "]"}:
                continue
            if '"content"' in candidate or candidate.lower().startswith("explicit"):
                continue
            candidate = candidate.strip("\"' ,")
            if len(candidate) < 12:
                continue
            observations.append(ExplicitObservationBase(content=candidate))
        return cls(explicit=observations)


class ExplicitObservation(ExplicitObservationBase, ObservationMetadata):
    """Explicit observation with content and metadata."""

    def __str__(self) -> str:
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] {self.content}"

    def str_with_id(self) -> str:
        id_prefix = f"[id:{self.id}] " if self.id else ""
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] {self.content}"

    def __hash__(self) -> int:
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExplicitObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class DeductiveObservation(DeductiveObservationBase, ObservationMetadata):
    """Deductive observation with multiple premises and one conclusion, plus metadata."""

    def __str__(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] {self.conclusion}\n{premises_text}"

    def str_with_id(self) -> str:
        id_prefix = f"[id:{self.id}] " if self.id else ""
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] {self.conclusion}\n{premises_text}"

    def str_no_timestamps(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class InductiveObservation(InductiveObservationBase, ObservationMetadata):
    """Inductive observation with sources, pattern type, and confidence, plus metadata."""

    def __str__(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] [{self.confidence}] {self.conclusion}{sources_text}"

    def str_with_id(self) -> str:
        id_prefix = f"[id:{self.id}] " if self.id else ""
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] [{self.confidence}] {self.conclusion}{sources_text}"

    def str_no_timestamps(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"[{self.confidence}] {self.conclusion}{sources_text}"

    def __hash__(self) -> int:
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class ContradictionObservation(ContradictionObservationBase, ObservationMetadata):
    def __str__(self) -> str:
        sources_text = "\n".join(f"    - {source}" for source in self.sources)
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] CONTRADICTION: {self.content}\n{sources_text}"

    def str_with_id(self) -> str:
        id_prefix = f"[id:{self.id}] " if self.id else ""
        sources_text = "\n".join(f"    - {source}" for source in self.sources)
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] CONTRADICTION: {self.content}\n{sources_text}"

    def __hash__(self) -> int:
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContradictionObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class Representation(BaseModel):
    """A Representation is a traversable and diffable map of observations."""

    explicit: list[ExplicitObservation] = Field(default_factory=list)
    deductive: list[DeductiveObservation] = Field(default_factory=list)
    inductive: list[InductiveObservation] = Field(default_factory=list)
    contradiction: list[ContradictionObservation] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.explicit or self.deductive or self.inductive or self.contradiction
        )

    def merge_representation(
        self, other: "Representation", max_observations: int | None = None
    ) -> None:
        merged_explicit = list(dict.fromkeys([*self.explicit, *other.explicit]))
        merged_deductive = list(dict.fromkeys([*self.deductive, *other.deductive]))
        merged_inductive = list(dict.fromkeys([*self.inductive, *other.inductive]))
        merged_contradiction = list(
            dict.fromkeys([*self.contradiction, *other.contradiction])
        )

        merged_explicit.sort(key=lambda obs: obs.created_at)
        merged_deductive.sort(key=lambda obs: obs.created_at)
        merged_inductive.sort(key=lambda obs: obs.created_at)
        merged_contradiction.sort(key=lambda obs: obs.created_at)

        if max_observations is not None:
            merged_explicit = merged_explicit[-max_observations:]
            merged_deductive = merged_deductive[-max_observations:]
            merged_inductive = merged_inductive[-max_observations:]
            merged_contradiction = merged_contradiction[-max_observations:]

        self.explicit = merged_explicit
        self.deductive = merged_deductive
        self.inductive = merged_inductive
        self.contradiction = merged_contradiction

    def format_as_markdown(self, include_ids: bool = False) -> str:
        return self.to_markdown(include_ids=include_ids)

    def to_markdown(self, include_ids: bool = False) -> str:
        parts: list[str] = []
        if self.explicit:
            parts.append("## Explicit Observations\n")
            for obs in self.explicit:
                parts.append(f"{obs}")
            parts.append("")
        if self.deductive:
            parts.append("## Deductive Observations\n")
            for obs in self.deductive:
                id_prefix = f"[id:{obs.id}] " if include_ids and obs.id else ""
                timestamp = _strip_microseconds_and_timezone(obs.created_at)
                parts.append(f"{id_prefix}[{timestamp}] {obs.conclusion}")
                if obs.premises:
                    parts.append("   Premises:")
                    for premise in obs.premises:
                        parts.append(f"   - {premise}")
                parts.append("")
            parts.append("")
        if self.inductive:
            parts.append("## Inductive Observations\n")
            for obs in self.inductive:
                id_prefix = f"[id:{obs.id}] " if include_ids and obs.id else ""
                parts.append(
                    f"{id_prefix} **Pattern** [{obs.confidence}]: {obs.conclusion}"
                )
                if obs.pattern_type:
                    parts.append(f"   **Type**: {obs.pattern_type}")
                if obs.sources:
                    parts.append("   **Sources**:")
                    for source in obs.sources[:5]:
                        parts.append(f"   - {source}")
                    if len(obs.sources) > 5:
                        parts.append(f"   - ... and {len(obs.sources) - 5} more")
                parts.append("")
            parts.append("")
        if self.contradiction:
            parts.append("## Contradictions\n")
            for obs in self.contradiction:
                id_prefix = f"[id:{obs.id}] " if include_ids and obs.id else ""
                parts.append(f"{id_prefix} **CONTRADICTION**: {obs.content}")
                if obs.sources:
                    parts.append("   **Conflicting statements**:")
                    for source in obs.sources:
                        parts.append(f"   - {source}")
                parts.append("")
            parts.append("")
        return "\n".join(parts)

    @classmethod
    def from_documents(cls, documents: Sequence[models.Document]) -> "Representation":
        return cls(
            explicit=[
                ExplicitObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=flatten_message_ids(
                        doc.internal_metadata.get("message_ids", [])
                    ),
                    session_name=doc.session_name,
                )
                for doc in documents
                if doc.level == "explicit"
            ],
            deductive=[
                DeductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=flatten_message_ids(
                        doc.internal_metadata.get("message_ids", [])
                    ),
                    session_name=doc.session_name,
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("premise_ids", []),
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.level == "deductive"
            ],
            inductive=[
                InductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("source_ids", []),
                    sources=doc.internal_metadata.get("sources", []),
                    pattern_type=doc.internal_metadata.get("pattern_type", "pattern"),
                    confidence=doc.internal_metadata.get("confidence", "medium"),
                )
                for doc in documents
                if doc.level == "inductive"
            ],
            contradiction=[
                ContradictionObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=flatten_message_ids(
                        doc.internal_metadata.get("message_ids", [])
                    ),
                    session_name=doc.session_name,
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("source_ids", []),
                    sources=doc.internal_metadata.get("sources", []),
                )
                for doc in documents
                if doc.level == "contradiction"
            ],
        )

    @classmethod
    def from_prompt_representation(
        cls,
        prompt_representation: "PromptRepresentation",
        message_ids: list[int],
        session_name: str,
        created_at: datetime,
    ) -> "Representation":
        return cls(
            explicit=[
                ExplicitObservation(
                    content=e.content,
                    created_at=created_at,
                    message_ids=message_ids,
                    session_name=session_name,
                )
                for e in prompt_representation.explicit
            ],
            deductive=[],
            inductive=[],
        )


def _safe_datetime_from_metadata(
    internal_metadata: dict[str, Any], fallback_datetime: datetime
) -> datetime:
    message_created_at = internal_metadata.get("message_created_at")
    if message_created_at is None:
        return _strip_microseconds_and_timezone(fallback_datetime)
    if isinstance(message_created_at, str):
        try:
            return _strip_microseconds_and_timezone(
                parse_datetime_iso(message_created_at)
            )
        except ValueError:
            return _strip_microseconds_and_timezone(fallback_datetime)
    if isinstance(message_created_at, datetime):
        return _strip_microseconds_and_timezone(message_created_at)
    return _strip_microseconds_and_timezone(fallback_datetime)
