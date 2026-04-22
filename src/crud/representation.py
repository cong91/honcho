from __future__ import annotations

import datetime
import logging
import time
from contextlib import suppress
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, exceptions, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.embedding_client import embedding_client
from src.schemas import ResolvedConfiguration
from src.telemetry.logging import accumulate_metric
from src.utils.formatting import format_datetime_utc
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)

logger = logging.getLogger(__name__)


def _memory_priority_score(document: models.Document, query: str | None = None) -> int:
    memory = (document.internal_metadata or {}).get("memory") or {}
    if not isinstance(memory, dict):
        return 0

    score = 0
    thesis_kind = memory.get("thesis_kind")
    horizon = memory.get("horizon")
    domain = str(memory.get("domain") or "").lower()
    query_lc = (query or "").lower()

    thesis_boost = {
        "rule": 40,
        "decision": 30,
        "preference": 25,
        "fact": 15,
        "plan": 10,
        "state": 5,
    }
    horizon_boost = {
        "long": 15,
        "medium": 8,
        "short": 2,
    }

    score += thesis_boost.get(thesis_kind, 0)
    score += horizon_boost.get(horizon, 0)

    if not query_lc:
        return score

    intent_token_groups = {
        "preference": ["prefer", "preference", "like", "want"],
        "rule": ["rule", "policy", "convention", "instruction"],
        "decision": ["decide", "decision", "architecture", "why"],
        "state": ["status", "current", "now", "blocker"],
        "plan": ["plan", "roadmap", "next", "timeline"],
        "fact": ["fact", "confirm", "what is", "where is"],
    }
    intent_boosts = {
        "preference": 20,
        "rule": 20,
        "decision": 20,
        "state": 35,
        "plan": 18,
        "fact": 12,
    }
    horizon_intent_boosts = {
        "state": {"short": 15, "medium": 6, "long": -10},
        "plan": {"medium": 10, "short": 4, "long": -4},
        "preference": {"long": 10, "medium": 3},
        "rule": {"long": 10, "medium": 3},
        "decision": {"medium": 8, "long": 6},
        "fact": {"long": 4, "medium": 2},
    }

    matching_intents = {
        intent
        for intent, tokens in intent_token_groups.items()
        if any(token in query_lc for token in tokens)
    }

    if thesis_kind in matching_intents:
        score += intent_boosts.get(thesis_kind, 0)

    for intent in matching_intents:
        score += horizon_intent_boosts.get(intent, {}).get(horizon, 0)

    domain_tokens = [
        token
        for token in domain.replace(":", "-").replace("_", "-").split("-")
        if token
    ]
    matched_domain_tokens = [token for token in domain_tokens if token in query_lc]
    if matched_domain_tokens:
        score += 8 * len(set(matched_domain_tokens))
        if len(set(matched_domain_tokens)) >= 2:
            score += 10

    return score


def _prioritize_documents(
    documents: list[models.Document], query: str | None = None
) -> list[models.Document]:
    return sorted(
        documents,
        key=lambda doc: (_memory_priority_score(doc, query), doc.created_at),
        reverse=True,
    )


class RepresentationManager:
    """Unified manager for representation and document queries."""

    def __init__(
        self,
        workspace_name: str,
        *,
        observer: str,
        observed: str,
    ) -> None:
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed

    async def save_representation(
        self,
        representation: Representation,
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        """
        Save Representation objects to the collection as a set of documents.

        Args:
            representation: Representation object
            message_ids: Message ID range to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created

        Returns:
            The number of *new documents saved*
        """

        new_documents = 0

        if not representation.deductive and not representation.explicit:
            logger.debug("No observations to save")
            return new_documents

        all_observations = representation.deductive + representation.explicit

        # Batch embed all observations
        batch_embed_start = time.perf_counter()

        observation_texts = [
            obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content
            for obs in all_observations
        ]
        try:
            embeddings = await embedding_client.simple_batch_embed(observation_texts)
        except ValueError as e:
            raise exceptions.ValidationException(
                "Observation content exceeds maximum token limit of "
                + f"{settings.EMBEDDING.MAX_INPUT_TOKENS}."
            ) from e

        batch_embed_duration = (time.perf_counter() - batch_embed_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "embed_new_observations",
            batch_embed_duration,
            "ms",
        )

        # Batch create document objects
        create_document_start = time.perf_counter()
        async with tracked_db("representation_manager.save_representation") as db:
            new_documents = await self._save_representation_internal(
                db,
                all_observations,
                embeddings,
                message_ids,
                session_name,
                message_created_at,
                message_level_configuration,
            )

        create_document_duration = (time.perf_counter() - create_document_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "save_new_observations",
            create_document_duration,
            "ms",
        )

        return new_documents

    async def _save_representation_internal(
        self,
        db: AsyncSession,
        all_observations: list[ExplicitObservation | DeductiveObservation],
        embeddings: list[list[float]],
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        # get_or_create_collection already handles IntegrityError with rollback and a retry
        collection = await crud.get_or_create_collection(
            db,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
        )

        # Prepare all documents for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs, embedding in zip(all_observations, embeddings, strict=True):
            # NOTE: will add additional levels of reasoning in the future
            if isinstance(obs, DeductiveObservation):
                obs_level = "deductive"
                obs_content = obs.conclusion
                obs_premises = obs.premises
            else:
                obs_level = "explicit"
                obs_content = obs.content
                obs_premises = None

            metadata: schemas.DocumentMetadata = schemas.DocumentMetadata(
                message_ids=message_ids,
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(
                    content=obs_content,
                    session_name=session_name,
                    level=obs_level,
                    metadata=metadata,
                    embedding=embedding,
                )
            )

        # Use bulk creation with optional duplicate detection
        accepted_documents = await crud.create_documents(
            db,
            documents_to_create,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            deduplicate=settings.DERIVER.DEDUPLICATE,
        )

        if message_level_configuration.dream.enabled:
            try:
                await check_and_schedule_dream(db, collection)
            except Exception as e:
                logger.warning(f"Failed to check dream scheduling: {e}")

        return len(accepted_documents)

    async def get_working_representation(
        self,
        *,
        db: AsyncSession | None = None,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        memory_domains: list[str] | None = None,
        memory_horizons: list[str] | None = None,
        memory_thesis_kinds: list[str] | None = None,
        exclude_expired: bool = True,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    ) -> Representation:
        """
        Get working representation with flexible query options.

        Args:
            db: Optional database session. If provided, uses it directly;
                otherwise creates a new session via tracked_db.
            session_name: Optional session to filter by
            include_semantic_query: Query for semantic search
            embedding: Pre-computed embedding for the semantic query.
            semantic_search_top_k: Number of semantic results
            semantic_search_max_distance: Maximum distance for semantic search
            include_most_derived: Include most derived observations
            memory_domains: Optional taxonomy domain filters
            memory_horizons: Optional taxonomy horizon filters
            memory_thesis_kinds: Optional taxonomy thesis kind filters
            exclude_expired: Exclude expired/review-due memory items
            max_observations: Maximum total observations to return

        Returns:
            Representation combining various query strategies
        """
        if include_semantic_query and embedding is None:
            with suppress(Exception):
                # Best-effort precompute
                embedding = await embedding_client.embed(include_semantic_query)

        if db is not None:
            return await self._get_working_representation_internal(
                db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                include_most_derived=include_most_derived,
                memory_domains=memory_domains,
                memory_horizons=memory_horizons,
                memory_thesis_kinds=memory_thesis_kinds,
                exclude_expired=exclude_expired,
                max_observations=max_observations,
            )

        async with tracked_db(
            "representation_manager.get_working_representation"
        ) as new_db:
            return await self._get_working_representation_internal(
                new_db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                include_most_derived=include_most_derived,
                memory_domains=memory_domains,
                memory_horizons=memory_horizons,
                memory_thesis_kinds=memory_thesis_kinds,
                exclude_expired=exclude_expired,
                max_observations=max_observations,
            )

    # Private helper methods

    async def _get_working_representation_internal(
        self,
        db: AsyncSession,
        *,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        memory_domains: list[str] | None = None,
        memory_horizons: list[str] | None = None,
        memory_thesis_kinds: list[str] | None = None,
        exclude_expired: bool = True,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    ) -> Representation:
        """Internal implementation of get_working_representation."""
        total = max_observations

        # Calculate how many observations to get from each source
        semantic_observations = (
            min(
                max(
                    0,
                    semantic_search_top_k
                    if semantic_search_top_k is not None
                    else total // 3,
                ),
                total,
            )
            if include_semantic_query
            else 0
        )

        if include_semantic_query and include_most_derived:
            # three-way blend: both semantic and derived requested
            top_observations = min(max(0, total // 3), total - semantic_observations)
        elif include_most_derived:
            # two-way blend: only derived requested
            top_observations = min(max(0, total // 2), total - semantic_observations)
        else:
            # no derived observations requested
            top_observations = 0

        # remaining observations are recent
        recent_observations = total - semantic_observations - top_observations

        representation = Representation()

        # Get semantic observations if requested
        if include_semantic_query:
            semantic_docs = await self._query_documents_semantic(
                db,
                query=include_semantic_query,
                top_k=semantic_observations,
                max_distance=semantic_search_max_distance,
                embedding=embedding,
                filters=self._build_filter_conditions(
                    memory_domains=memory_domains,
                    memory_horizons=memory_horizons,
                    memory_thesis_kinds=memory_thesis_kinds,
                ),
            )
            representation.merge_representation(
                Representation.from_documents(semantic_docs)
            )

        # Get most derived observations if requested
        if include_most_derived:
            derived_docs = await self._query_documents_most_derived(
                db,
                top_k=top_observations,
                filters=self._build_filter_conditions(
                    memory_domains=memory_domains,
                    memory_horizons=memory_horizons,
                    memory_thesis_kinds=memory_thesis_kinds,
                ),
                exclude_expired=exclude_expired,
            )
            representation.merge_representation(
                Representation.from_documents(derived_docs)
            )

        # Get recent observations
        recent_docs = await self._query_documents_recent(
            db,
            top_k=recent_observations,
            session_name=session_name,
            filters=self._build_filter_conditions(
                memory_domains=memory_domains,
                memory_horizons=memory_horizons,
                memory_thesis_kinds=memory_thesis_kinds,
            ),
            exclude_expired=exclude_expired,
        )

        representation.merge_representation(Representation.from_documents(recent_docs))

        return representation

    async def _query_documents_semantic(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float | None = None,
        level: str | None = None,
        embedding: list[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[models.Document]:
        """Query documents by semantic similarity."""
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    top_k,
                    max_distance,
                    embedding=embedding,
                    filters=filters,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=query,
                    filters=filters,
                    max_distance=max_distance,
                    top_k=top_k,
                    embedding=embedding,
                )
                db.expunge_all()
                return _prioritize_documents(list(documents), query)

        except Exception as e:
            logger.error(f"Error getting relevant observations: {e}")
            return []

    async def _query_documents_recent(
        self,
        db: AsyncSession,
        top_k: int,
        session_name: str | None = None,
        filters: dict[str, Any] | None = None,
        exclude_expired: bool = True,
    ) -> list[models.Document]:
        """Query most recent documents."""
        documents = await crud.query_documents_recent(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            limit=top_k,
            session_name=session_name,
            filters=filters,
            exclude_expired=exclude_expired,
        )
        db.expunge_all()
        return _prioritize_documents(list(documents))

    async def _query_documents_most_derived(
        self,
        db: AsyncSession,
        top_k: int,
        filters: dict[str, Any] | None = None,
        exclude_expired: bool = True,
    ) -> list[models.Document]:
        """Query most derived documents."""
        documents = await crud.query_documents_most_derived(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            limit=top_k,
            filters=filters,
            exclude_expired=exclude_expired,
        )
        db.expunge_all()
        return _prioritize_documents(list(documents))

    async def _get_observations_internal(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float,
        level: str | None,
    ) -> list[models.Document]:
        """Internal method that does the actual observation retrieval."""
        return await self._query_documents_semantic(
            db, query, top_k, max_distance, level
        )

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        count: int,
        max_distance: float | None = None,
        embedding: list[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=query,
            max_distance=max_distance,
            top_k=count,
            filters=self._build_filter_conditions(
                level,
                memory_domains=(filters or {}).get("metadata.memory.domain", {}).get("in")
                if isinstance((filters or {}).get("metadata.memory.domain"), dict)
                else None,
                memory_horizons=(filters or {}).get("metadata.memory.horizon", {}).get("in")
                if isinstance((filters or {}).get("metadata.memory.horizon"), dict)
                else None,
                memory_thesis_kinds=(filters or {}).get("metadata.memory.thesis_kind", {}).get("in")
                if isinstance((filters or {}).get("metadata.memory.thesis_kind"), dict)
                else None,
            ),
            embedding=embedding,
        )

        # Sort by creation time
        docs_sorted: list[models.Document] = sorted(
            list(documents), key=lambda x: x.created_at, reverse=True
        )
        return docs_sorted

    def _build_filter_conditions(
        self,
        level: str | None = None,
        memory_domains: list[str] | None = None,
        memory_horizons: list[str] | None = None,
        memory_thesis_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build filter conditions for document queries.

        Returns a flat dict of key-value pairs for vector store filtering.
        """
        filters: dict[str, Any] = {}

        if level:
            filters["level"] = level
        if memory_domains:
            filters["metadata.memory.domain"] = {"in": memory_domains}
        if memory_horizons:
            filters["metadata.memory.horizon"] = {"in": memory_horizons}
        if memory_thesis_kinds:
            filters["metadata.memory.thesis_kind"] = {"in": memory_thesis_kinds}

        return filters


# Module-level functions for backward compatibility and convenience


async def get_working_representation(
    workspace_name: str,
    *,
    db: AsyncSession | None = None,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    embedding: list[float] | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    include_most_derived: bool = False,
    memory_domains: list[str] | None = None,
    memory_horizons: list[str] | None = None,
    memory_thesis_kinds: list[str] | None = None,
    exclude_expired: bool = True,
    max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
) -> Representation:
    """
    Get raw working representation data from the relevant document collection.

    This is a convenience function that creates a RepresentationManager and calls
    get_working_representation on it.

    Args:
        db: Optional database session. If provided, uses it directly;
            otherwise creates a new session via tracked_db.
        embedding: Pre-computed embedding for the semantic query.
    """
    manager = RepresentationManager(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    return await manager.get_working_representation(
        db=db,
        session_name=session_name,
        include_semantic_query=include_semantic_query,
        embedding=embedding,
        semantic_search_top_k=semantic_search_top_k,
        semantic_search_max_distance=semantic_search_max_distance,
        include_most_derived=include_most_derived,
        memory_domains=memory_domains,
        memory_horizons=memory_horizons,
        memory_thesis_kinds=memory_thesis_kinds,
        exclude_expired=exclude_expired,
        max_observations=max_observations,
    )
