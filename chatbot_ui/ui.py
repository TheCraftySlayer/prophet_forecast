"""Interactive command line interface for the CustomGPT agent orchestrator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .client import DEFAULT_BASE_URL, CustomGPTClient
from .orchestrator import AgentConfig, AgentOrchestrator

console = Console()


@dataclass(frozen=True)
class AgentDefinition:
    key: str
    name: str
    project_id: int
    description: str


DEFAULT_AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        key="cartographer",
        name="A.C.E Assessor’s Cartographic Explorer",
        project_id=83668,
        description="Geospatial insights for assessor parcels and zoning.",
    ),
    AgentDefinition(
        key="expectations",
        name="A.C.E (Advisor for Clear Expectations)",
        project_id=49501,
        description="Clarifies policy expectations and assessment criteria.",
    ),
    AgentDefinition(
        key="compliance",
        name="A.C.E (Assessor's Compliance Expert)",
        project_id=37400,
        description="Guides compliance workflows and regulatory checks.",
    ),
    AgentDefinition(
        key="community",
        name="A.C.E (Assessor's Community Educator)",
        project_id=9262,
        description="Translates assessor decisions for the public and stakeholders.",
    ),
)


def build_orchestrator(api_key: str, *, base_url: Optional[str] = None) -> AgentOrchestrator:
    """Factory that builds an :class:`AgentOrchestrator` with default agents."""

    configured_base = base_url or os.getenv("CUSTOMGPT_BASE_URL") or DEFAULT_BASE_URL
    client = CustomGPTClient(api_key, base_url=configured_base)
    agents = (
        AgentConfig(defn.key, defn.name, defn.project_id, defn.description)
        for defn in DEFAULT_AGENTS
    )
    return AgentOrchestrator(client, agents)


def render_agent_table(orchestrator: AgentOrchestrator) -> None:
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Key", style="bold cyan")
    table.add_column("Agent")
    table.add_column("Project ID", justify="right")
    table.add_column("Description")

    for agent in orchestrator.list_agents():
        table.add_row(agent.key, agent.name, str(agent.project_id), agent.description)

    console.print(table)


def run_chat(orchestrator: AgentOrchestrator, *, default_agent: Optional[str] = None) -> None:
    if default_agent is None:
        render_agent_table(orchestrator)
        agent_key = Prompt.ask(
            "Select an agent to chat with",
            choices=list(orchestrator.agents.keys()),
            default=next(iter(orchestrator.agents)),
        )
    else:
        agent_key = default_agent

    console.print(f"\n[bold green]Chatting with {orchestrator.agents[agent_key].name}[/bold green]\n")
    console.print("Type [bold]/switch[/bold] to change agents, [bold]/reset[/bold] to clear history, or [bold]/exit[/bold] to quit.\n")

    while True:
        user_input = Prompt.ask("You")

        if not user_input.strip():
            continue

        if user_input.strip().lower() == "/exit":
            console.print("Goodbye! 👋")
            break

        if user_input.strip().lower() == "/switch":
            render_agent_table(orchestrator)
            agent_key = Prompt.ask(
                "Switch to which agent?",
                choices=list(orchestrator.agents.keys()),
                default=agent_key,
            )
            console.print(
                f"\n[bold green]Now chatting with {orchestrator.agents[agent_key].name}[/bold green]\n"
            )
            continue

        if user_input.strip().lower() == "/reset":
            orchestrator.reset(agent_key)
            console.print("Conversation history cleared. Start fresh!")
            continue

        response = orchestrator.send(agent_key, user_input)
        _render_response(agent_key, response.text)


def _render_response(agent_key: str, text: str) -> None:
    panel = Panel.fit(
        text,
        title=f"Agent: {agent_key}",
        title_align="left",
        border_style="magenta",
    )
    console.print(panel)


def main(argv: Optional[Iterable[str]] = None) -> None:
    api_key = os.getenv("CUSTOMGPT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the CUSTOMGPT_API_KEY environment variable before launching the chat UI."
        )

    orchestrator = build_orchestrator(api_key)
    run_chat(orchestrator)


if __name__ == "__main__":
    main()
