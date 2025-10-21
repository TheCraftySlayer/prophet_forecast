from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from unittest.mock import Mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from chatbot_ui.client import ChatResponse, CustomGPTClient
from chatbot_ui.orchestrator import AgentConfig, AgentOrchestrator


def _build_mock_response(response_body: dict) -> Mock:
    response = Mock()
    response.json.return_value = response_body
    response.raise_for_status.return_value = None
    return response


def test_client_send_message_constructs_request() -> None:
    session = Mock()
    session.post.return_value = _build_mock_response(
        {"response": "Hi there!", "conversation_id": "conv-123"}
    )

    client = CustomGPTClient(
        "test-key",
        base_url="https://example.com/api",
        session=session,
    )

    result = client.send_message(42, "Hello")

    session.post.assert_called_once_with(
        "https://example.com/api/projects/42/conversations",
        json={"input": "Hello"},
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    assert result.text == "Hi there!"
    assert result.conversation_id == "conv-123"
    assert result.raw["response"] == "Hi there!"


def test_client_includes_conversation_id() -> None:
    session = Mock()
    session.post.return_value = _build_mock_response({"response": "Continue"})

    client = CustomGPTClient("key", session=session)
    client.send_message(999, "follow up", conversation_id="thread-001")

    payload = session.post.call_args.kwargs["json"]
    assert payload == {"input": "follow up", "conversation_id": "thread-001"}


@dataclass
class StubAgent:
    key: str
    name: str
    project_id: int
    description: str


class StubClient:
    def __init__(self, responses: Iterable[ChatResponse]):
        self._responses = iter(responses)
        self.calls: list[tuple[int, str, str | None]] = []

    def send_message(
        self, project_id: int, message: str, *, conversation_id: str | None = None
    ) -> ChatResponse:
        self.calls.append((project_id, message, conversation_id))
        return next(self._responses)


def test_orchestrator_tracks_conversations() -> None:
    responses = [
        ChatResponse(text="First reply", conversation_id="alpha", raw={}),
        ChatResponse(text="Second reply", conversation_id="alpha", raw={}),
    ]
    orchestrator = AgentOrchestrator(
        StubClient(responses),
        [AgentConfig("a", "Agent A", 1, "First agent")],
    )

    first = orchestrator.send("a", "Hello")
    second = orchestrator.send("a", "How are you?")

    assert first.text == "First reply"
    assert second.text == "Second reply"

    conversation = orchestrator.get_conversation("a")
    assert conversation.conversation_id == "alpha"
    assert conversation.messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Second reply"},
    ]


def test_orchestrator_reset_clears_history() -> None:
    responses = [ChatResponse(text="Reply", conversation_id="beta", raw={})]
    client = StubClient(responses)
    orchestrator = AgentOrchestrator(
        client,
        [AgentConfig("b", "Agent B", 2, "Second agent")],
    )

    orchestrator.send("b", "Ping")
    orchestrator.reset("b")

    conversation = orchestrator.get_conversation("b")
    assert conversation.conversation_id is None
    assert conversation.messages == []


def test_orchestrator_missing_agent_raises_error() -> None:
    client = StubClient(
        [ChatResponse(text="unused", conversation_id=None, raw={})]
    )
    orchestrator = AgentOrchestrator(
        client,
        [AgentConfig("c", "Agent C", 3, "Third agent")],
    )

    with pytest.raises(KeyError):
        orchestrator.send("unknown", "Hello")
