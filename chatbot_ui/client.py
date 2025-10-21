"""Client utilities for communicating with the CustomGPT API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - compatibility shim
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    class _FallbackSession:
        """Minimal stub used when the real requests package is unavailable."""

        def post(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                "The requests package is required to send HTTP requests. Install it or "
                "provide a custom session object to CustomGPTClient."
            )

    class _RequestsModule:  # pragma: no cover - shim container
        Session = _FallbackSession

    requests = _RequestsModule()  # type: ignore[assignment]


DEFAULT_BASE_URL = "https://api.customgpt.ai/v1"


@dataclass(slots=True)
class ChatResponse:
    """Container for responses returned by the CustomGPT API."""

    text: str
    conversation_id: Optional[str]
    raw: Dict[str, Any]


class CustomGPTClient:
    """Minimal API client that wraps the CustomGPT chat endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def send_message(
        self, project_id: int | str, message: str, *, conversation_id: Optional[str] = None
    ) -> ChatResponse:
        """Send a message to a project and return the model response."""

        url = f"{self.base_url}/projects/{project_id}/conversations"
        payload: Dict[str, Any] = {"input": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self.session.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

        text = _extract_response_text(data)
        conversation_identifier = _extract_conversation_id(data) or conversation_id

        return ChatResponse(text=text, conversation_id=conversation_identifier, raw=data)


def _extract_response_text(data: Dict[str, Any]) -> str:
    """Best-effort extraction of the assistant response."""

    if "response" in data and isinstance(data["response"], str):
        return data["response"]

    if "data" in data:
        nested = data["data"]
        if isinstance(nested, dict):
            if "response" in nested and isinstance(nested["response"], str):
                return nested["response"]
            if "output" in nested and isinstance(nested["output"], str):
                return nested["output"]

    if "message" in data and isinstance(data["message"], str):
        return data["message"]

    raise ValueError("Unable to parse response text from CustomGPT response payload")


def _extract_conversation_id(data: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the conversation identifier."""

    for key in ("conversation_id", "conversationId", "id"):
        value = data.get(key)
        if isinstance(value, str):
            return value

    if "data" in data and isinstance(data["data"], dict):
        nested = data["data"]
        for key in ("conversation_id", "conversationId", "id"):
            value = nested.get(key)
            if isinstance(value, str):
                return value

    return None
