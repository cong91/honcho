import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.llm.types import HonchoLLMCallResponse
from src.models import Peer, Workspace
from src.utils.config_helpers import get_configuration
from src.utils.representation import ExplicitObservationBase, PromptRepresentation


async def create_test_session_with_peer(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
) -> models.Session:
    session = (
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={peer.name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            workspace.name,
        )
    ).resource
    await db_session.commit()
    return session


async def create_test_messages(
    db_session: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str,
    count: int = 1,
    content_prefix: str = "Test message",
) -> list[models.Message]:
    messages: list[models.Message] = []
    for i in range(count):
        message = models.Message(
            workspace_name=workspace_name,
            session_name=session_name,
            peer_name=peer_name,
            content=f"{content_prefix} {i}",
            public_id=generate_nanoid(),
            seq_in_session=i + 1,
            token_count=10,
        )
        db_session.add(message)
        messages.append(message)

    await db_session.commit()
    for msg in messages:
        await db_session.refresh(msg)
    return messages


def create_structured_deriver_response(
    content: PromptRepresentation,
    *,
    output_tokens: int = 42,
) -> HonchoLLMCallResponse[PromptRepresentation]:
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=100,
        output_tokens=output_tokens,
        finish_reasons=["end_turn"],
    )


@pytest.mark.asyncio
class TestDeriverStructuredOutputStability:
    async def test_deriver_retries_with_schema_only_when_json_mode_returns_empty(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        from src.deriver.deriver import process_representation_tasks_batch

        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session,
            workspace.name,
            session.name,
            peer.name,
            count=1,
            content_prefix="I prefer technical collaboration in Vietnamese.",
        )
        message_config = get_configuration(None, session, workspace)

        empty_response = create_structured_deriver_response(PromptRepresentation(explicit=[]))
        recovered_response = create_structured_deriver_response(
            PromptRepresentation(
                explicit=[
                    ExplicitObservationBase(
                        content="User prefers technical collaboration in Vietnamese."
                    )
                ]
            )
        )

        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(side_effect=[empty_response, recovered_response]),
            ) as mock_llm_call,
            patch(
                "src.crud.representation.RepresentationManager.save_representation",
                new=AsyncMock(),
            ) as mock_save,
        ):
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=message_config,
                observers=[peer.name],
                observed=peer.name,
                queue_item_message_ids=[m.id for m in messages],
            )

        assert mock_llm_call.await_count == 2
        first_kwargs = mock_llm_call.await_args_list[0].kwargs
        second_kwargs = mock_llm_call.await_args_list[1].kwargs
        assert first_kwargs["json_mode"] is True
        assert second_kwargs.get("json_mode", False) is False
        assert second_kwargs["response_model"] is PromptRepresentation
        mock_save.assert_awaited_once()

    async def test_deriver_logs_raw_payload_on_empty_after_retry(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        caplog: pytest.LogCaptureFixture,
    ):
        from src.deriver.deriver import process_representation_tasks_batch

        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session,
            workspace.name,
            session.name,
            peer.name,
            count=1,
            content_prefix="I work remotely from Hanoi.",
        )
        message_config = get_configuration(None, session, workspace)

        empty_response = create_structured_deriver_response(PromptRepresentation(explicit=[]))

        caplog.set_level("WARNING")

        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(side_effect=[empty_response, empty_response]),
            ),
            patch(
                "src.crud.representation.RepresentationManager.save_representation",
                new=AsyncMock(),
            ) as mock_save,
        ):
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=message_config,
                observers=[peer.name],
                observed=peer.name,
                queue_item_message_ids=[m.id for m in messages],
            )

        mock_save.assert_not_awaited()
        assert any(
            "Deriver generated zero observations" in record.message
            and "raw_output=" in record.message
            for record in caplog.records
        )
