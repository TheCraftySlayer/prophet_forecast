"""Core orchestration primitives for coordinating CustomGPT agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .client import ChatResponse, CustomGPTClient


@dataclass(slots=True)
class AgentConfig:
    """Metadata describing an individual expert agent."""

    key: str
    name: str
    project_id: int
    description: str


@dataclass
class Conversation:
    """Stores conversation history for a single agent."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    conversation_id: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class AgentOrchestrator:
    """Coordinates the routing of chat messages to multiple CustomGPT agents."""

    def __init__(self, client: CustomGPTClient, agents: Iterable[AgentConfig]):
        self.client = client
        self.agents: Dict[str, AgentConfig] = {agent.key: agent for agent in agents}
        if not self.agents:
            raise ValueError("At least one agent configuration must be provided")
        self._conversations: Dict[str, Conversation] = {
            key: Conversation() for key in self.agents
        }

    def list_agents(self) -> List[AgentConfig]:
        """Return the list of configured agents."""

        return list(self.agents.values())

    def get_conversation(self, agent_key: str) -> Conversation:
        """Return the conversation object for the requested agent."""

        try:
            return self._conversations[agent_key]
        except KeyError as exc:
            raise KeyError(f"Agent '{agent_key}' is not configured") from exc

    def send(self, agent_key: str, message: str) -> ChatResponse:
        """Send a chat message to the selected agent."""

        agent = self._get_agent(agent_key)
        conversation = self.get_conversation(agent_key)
        conversation.add_turn("user", message)

        response = self.client.send_message(
            agent.project_id,
            message,
            conversation_id=conversation.conversation_id,
        )

        conversation.conversation_id = response.conversation_id
        conversation.add_turn("assistant", response.text)
        return response

    def reset(self, agent_key: str) -> None:
        """Clear the stored conversation for a given agent."""

        self._get_agent(agent_key)
        self._conversations[agent_key] = Conversation()

    def _get_agent(self, agent_key: str) -> AgentConfig:
        try:
            return self.agents[agent_key]
        except KeyError as exc:
            raise KeyError(f"Agent '{agent_key}' is not configured") from exc
