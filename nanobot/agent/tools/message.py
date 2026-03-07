"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        # Track sent state per session (session_key -> turn_id -> bool)
        # This is needed because MessageTool is a shared instance across concurrent sessions
        self._sent_in_session: dict[str, dict[str, bool]] = {}

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self, session_key: str) -> None:
        """Reset per-turn send tracking for a specific session."""
        if session_key in self._sent_in_session:
            self._sent_in_session[session_key] = {}

    def sent_in_turn(self, session_key: str, turn_id: str) -> bool:
        """Check if a message was already sent in this turn for this session."""
        return self._sent_in_session.get(session_key, {}).get(turn_id, False)

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        context: ToolContext | None = None,
        **kwargs: Any
    ) -> str:
        # Use context parameter if provided, otherwise fall back to global state
        if context is not None:
            target_channel = channel or context.channel
            target_chat_id = chat_id or context.chat_id
            target_message_id = message_id or context.message_id
            # Build session key from context for state tracking
            session_key = f"{context.channel}:{context.chat_id}"
            turn_id = context.turn_id
        else:
            target_channel = channel or self._default_channel
            target_chat_id = chat_id or self._default_chat_id
            target_message_id = message_id or self._default_message_id
            # Fall back to default session key
            session_key = f"{self._default_channel}:{self._default_chat_id}"
            turn_id = "default"

        if not target_channel or not target_chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=target_channel,
            chat_id=target_chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": target_message_id,
            }
        )

        try:
            await self._send_callback(msg)
            # Track if we sent to the default/current context (per session, per turn)
            if context is not None:
                if target_channel == context.channel and target_chat_id == context.chat_id:
                    if session_key not in self._sent_in_session:
                        self._sent_in_session[session_key] = {}
                    self._sent_in_session[session_key][turn_id] = True
            else:
                if target_channel == self._default_channel and target_chat_id == self._default_chat_id:
                    if session_key not in self._sent_in_session:
                        self._sent_in_session[session_key] = {}
                    self._sent_in_session[session_key][turn_id] = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {target_channel}:{target_chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
