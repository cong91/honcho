from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.llm.types import HonchoLLMCallResponse
from src.models import Peer, Workspace
from src.utils.config_helpers import get_configuration
from src.utils.representation import PromptRepresentation


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
    *,
    contents: list[str],
) -> list[models.Message]:
    messages: list[models.Message] = []
    for i, content in enumerate(contents, start=1):
        message = models.Message(
            workspace_name=workspace_name,
            session_name=session_name,
            peer_name=peer_name,
            content=content,
            public_id=generate_nanoid(),
            seq_in_session=i,
            token_count=20,
        )
        db_session.add(message)
        messages.append(message)

    await db_session.commit()
    for msg in messages:
        await db_session.refresh(msg)
    return messages


def create_response(content) -> HonchoLLMCallResponse:
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=100,
        output_tokens=42,
        finish_reasons=["end_turn"],
    )


@pytest.mark.asyncio
class TestDeriverPlaintextFallback:
    async def test_prompt_representation_from_plain_text_extracts_bullets(self):
        text = """
        - User prefers Vietnamese for technical collaboration.
        - User wants runtime-verified evidence during production debugging.
        """

        parsed = PromptRepresentation.from_plain_text(text)

        assert [obs.content for obs in parsed.explicit] == [
            "User prefers Vietnamese for technical collaboration.",
            "User wants runtime-verified evidence during production debugging.",
        ]

    async def test_deriver_saves_representation_from_plaintext_fallback(
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
            contents=[
                "I prefer technical collaboration in Vietnamese.",
                "I want runtime-verified evidence during production debugging.",
            ],
        )
        message_config = get_configuration(None, session, workspace)

        empty_structured = create_response(PromptRepresentation(explicit=[]))
        plaintext_retry = create_response(
            "- User prefers technical collaboration in Vietnamese.\n"
            "- User wants runtime-verified evidence during production debugging."
        )

        caplog.set_level("INFO")

        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(side_effect=[empty_structured, plaintext_retry]),
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
        mock_save.assert_awaited_once()
        saved_representation = mock_save.await_args.args[0]
        assert [obs.content for obs in saved_representation.explicit] == [
            "User prefers technical collaboration in Vietnamese.",
            "User wants runtime-verified evidence during production debugging.",
        ]
        assert any(
            "plain-text fallback" in record.message for record in caplog.records
        )
