# tests/test_tool_context.py
"""Tests for ToolContext dataclass."""

import pytest
from nanobot.agent.tools.context import ToolContext


class TestToolContext:
    def test_create_basic_context(self):
        """Test creating a basic ToolContext."""
        ctx = ToolContext(channel="telegram", chat_id="chat-123")
        assert ctx.channel == "telegram"
        assert ctx.chat_id == "chat-123"
        assert ctx.message_id is None
        assert ctx.turn_id is not None  # auto-generated

    def test_context_with_message_id(self):
        """Test creating context with message_id."""
        ctx = ToolContext(channel="discord", chat_id="c1", message_id="msg-456")
        assert ctx.message_id == "msg-456"

    def test_turn_id_uniqueness(self):
        """Test that each context gets a unique turn_id."""
        ctx1 = ToolContext(channel="telegram", chat_id="c1")
        ctx2 = ToolContext(channel="telegram", chat_id="c1")
        assert ctx1.turn_id != ctx2.turn_id

    def test_context_immutability(self):
        """Test that context is frozen (immutable)."""
        ctx = ToolContext(channel="telegram", chat_id="c1")
        with pytest.raises(AttributeError):
            ctx.channel = "discord"  # type: ignore
