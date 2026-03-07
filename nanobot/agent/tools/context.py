# nanobot/agent/tools/context.py
"""Tool execution context for routing and isolation."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class ToolContext:
    """Immutable context passed to tool executions for routing and isolation.

    Attributes:
        channel: The chat channel (e.g., "telegram", "discord")
        chat_id: The chat/conversation identifier
        message_id: Optional message ID for reply-to functionality
        turn_id: Unique identifier for this agent turn (auto-generated)
    """
    channel: str
    chat_id: str
    message_id: str | None = None
    turn_id: str = field(default_factory=lambda: str(uuid4())[:8])
