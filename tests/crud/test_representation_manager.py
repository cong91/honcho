import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.representation import (
    RepresentationManager,
    _prioritize_documents,
)


class TestRepresentationManagerSoftDelete:
    """Tests that RepresentationManager query methods exclude soft-deleted documents."""

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Collection, RepresentationManager]:
        """Create peers, session, collection, and a RepresentationManager."""
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        manager = RepresentationManager(
            test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        return test_peer2, test_session, collection, manager

    @pytest.mark.asyncio
    async def test_query_documents_recent_excludes_soft_deleted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Soft-deleted documents must not appear in the recent-documents query."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _, manager = await self._setup(
            db_session, test_workspace, test_peer
        )

        # Create two documents
        doc_live = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Live observation",
            session_name=test_session.name,
        )
        doc_deleted = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Deleted observation",
            session_name=test_session.name,
        )
        db_session.add_all([doc_live, doc_deleted])
        await db_session.flush()

        # Soft-delete one
        await db_session.execute(
            update(models.Document)
            .where(models.Document.id == doc_deleted.id)
            .values(deleted_at=func.now())
        )
        await db_session.commit()

        results = await manager._query_documents_recent(db_session, top_k=10)  # pyright: ignore[reportPrivateUsage]

        result_ids = [doc.id for doc in results]
        assert doc_live.id in result_ids
        assert doc_deleted.id not in result_ids

    @pytest.mark.asyncio
    async def test_query_documents_most_derived_excludes_soft_deleted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Soft-deleted documents must not appear in the most-derived query."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _, manager = await self._setup(
            db_session, test_workspace, test_peer
        )

        # Create two documents with different times_derived
        doc_live = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Live observation",
            session_name=test_session.name,
            times_derived=5,
        )
        doc_deleted = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Deleted high-derived observation",
            session_name=test_session.name,
            times_derived=100,
        )
        db_session.add_all([doc_live, doc_deleted])
        await db_session.flush()

        # Soft-delete the high-derived one
        await db_session.execute(
            update(models.Document)
            .where(models.Document.id == doc_deleted.id)
            .values(deleted_at=func.now())
        )
        await db_session.commit()

        results = await manager._query_documents_most_derived(db_session, top_k=10)  # pyright: ignore[reportPrivateUsage]

        result_ids = [doc.id for doc in results]
        assert doc_live.id in result_ids
        assert doc_deleted.id not in result_ids


class TestRepresentationMemoryPrioritization:
    def test_prioritize_documents_prefers_matching_taxonomy(self):
        preference_doc = models.Document(
            id="preference-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="User prefers concise replies",
            internal_metadata={
                "memory": {
                    "domain": "user:preferences",
                    "horizon": "long",
                    "thesis_kind": "preference",
                    "expiry": {"type": "none"},
                }
            },
        )
        state_doc = models.Document(
            id="state-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="Current blocker is missing config",
            internal_metadata={
                "memory": {
                    "domain": "project:current-state",
                    "horizon": "short",
                    "thesis_kind": "state",
                    "expiry": {"type": "none"},
                }
            },
        )

        prioritized = _prioritize_documents(
            [state_doc, preference_doc], query="what does the user prefer"
        )

        assert [doc.id for doc in prioritized] == ["preference-doc", "state-doc"]

    def test_prioritize_documents_prefers_current_state_for_status_queries(self):
        rule_doc = models.Document(
            id="rule-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="Workspace policy requires approvals before deploy",
            internal_metadata={
                "memory": {
                    "domain": "workspace:rule",
                    "horizon": "long",
                    "thesis_kind": "rule",
                    "expiry": {"type": "none"},
                }
            },
        )
        state_doc = models.Document(
            id="state-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="Current blocker is database migration failure",
            internal_metadata={
                "memory": {
                    "domain": "project:current-state",
                    "horizon": "short",
                    "thesis_kind": "state",
                    "expiry": {"type": "review", "review_at": "2026-04-22T01:00:00Z"},
                }
            },
        )

        prioritized = _prioritize_documents(
            [rule_doc, state_doc], query="what is the current blocker right now"
        )

        assert [doc.id for doc in prioritized] == ["state-doc", "rule-doc"]

    def test_prioritize_documents_prefers_domain_match_for_project_decisions(self):
        project_decision_doc = models.Document(
            id="project-decision-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="The payment service will use Stripe as the provider",
            internal_metadata={
                "memory": {
                    "domain": "project:payments",
                    "horizon": "medium",
                    "thesis_kind": "decision",
                    "expiry": {"type": "none"},
                }
            },
        )
        generic_decision_doc = models.Document(
            id="generic-decision-doc",
            workspace_name="workspace-a",
            observer="observer-a",
            observed="observed-b",
            content="The team chose a new internal tool",
            internal_metadata={
                "memory": {
                    "domain": "workspace:operations",
                    "horizon": "medium",
                    "thesis_kind": "decision",
                    "expiry": {"type": "none"},
                }
            },
        )

        prioritized = _prioritize_documents(
            [generic_decision_doc, project_decision_doc],
            query="why did we decide the payments architecture this way",
        )

        assert [doc.id for doc in prioritized] == [
            "project-decision-doc",
            "generic-decision-doc",
        ]
