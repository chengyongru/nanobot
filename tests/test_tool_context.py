# tests/test_tool_context.py
"""Tests for ToolContext dataclass."""

import asyncio
import inspect

import pytest
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


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


class MockToolWithContext(Tool):
    """Mock tool that accepts context parameter."""

    @property
    def name(self) -> str:
        return "mock_with_context"

    @property
    def description(self) -> str:
        return "A mock tool that uses context"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "An argument"}
            },
            "required": ["arg"],
        }

    async def execute(self, arg: str, context=None) -> str:
        if context:
            return f"got: {arg}, channel: {context.channel}"
        return f"got: {arg}, channel: none"


class MockToolWithoutContext(Tool):
    """Mock tool that does NOT accept context parameter."""

    @property
    def name(self) -> str:
        return "mock_no_context"

    @property
    def description(self) -> str:
        return "A mock tool without context"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "A value"}
            },
            "required": ["value"],
        }

    async def execute(self, value: str) -> str:
        return f"processed: {value}"


class TestToolRegistryContext:
    """Tests for ToolRegistry context support."""

    def test_execute_accepts_context(self):
        """Test that registry.execute() accepts optional context."""
        registry = ToolRegistry()
        registry.register(MockToolWithContext())

        ctx = ToolContext(channel="test", chat_id="c1")
        result = asyncio.run(registry.execute("mock_with_context", {"arg": "hello"}, context=ctx))
        assert "test" in result
        assert "hello" in result

    def test_execute_without_context(self):
        """Test that registry.execute() works without context."""
        registry = ToolRegistry()
        registry.register(MockToolWithContext())

        result = asyncio.run(registry.execute("mock_with_context", {"arg": "hello"}))
        assert "hello" in result
        assert "channel: none" in result

    def test_backward_compatibility_tool_without_context_param(self):
        """Test that tools without context param still work when context is passed."""
        registry = ToolRegistry()
        registry.register(MockToolWithoutContext())

        ctx = ToolContext(channel="telegram", chat_id="c1")
        # Should not raise error even though tool doesn't accept context
        result = asyncio.run(
            registry.execute("mock_no_context", {"value": "test"}, context=ctx)
        )
        assert "processed: test" in result

    def test_registry_passes_context_to_supporting_tools(self):
        """Test that context is properly passed to tools that support it."""
        registry = ToolRegistry()
        registry.register(MockToolWithContext())

        ctx = ToolContext(channel="discord", chat_id="chat-789", message_id="msg-123")
        result = asyncio.run(
            registry.execute("mock_with_context", {"arg": "data"}, context=ctx)
        )
        assert "discord" in result
        assert "data" in result


class TestMessageToolContext:
    """Tests for MessageTool context support."""

    @pytest.mark.asyncio
    async def test_message_tool_uses_context(self):
        """Test that MessageTool uses context parameter for routing."""
        from nanobot.agent.tools.message import MessageTool
        from nanobot.bus.events import OutboundMessage

        sent_messages = []

        async def mock_send(msg):
            sent_messages.append(msg)

        tool = MessageTool(send_callback=mock_send)
        ctx = ToolContext(channel="telegram", chat_id="chat-123")

        await tool.execute(content="Hello", context=ctx)

        assert len(sent_messages) == 1
        assert sent_messages[0].channel == "telegram"
        assert sent_messages[0].chat_id == "chat-123"

    @pytest.mark.asyncio
    async def test_message_tool_fallback_to_global_state(self):
        """Test backward compatibility: fall back to global state if no context."""
        from nanobot.agent.tools.message import MessageTool

        sent_messages = []

        async def mock_send(msg):
            sent_messages.append(msg)

        tool = MessageTool(send_callback=mock_send)
        tool.set_context("discord", "channel-456")  # Set global state

        await tool.execute(content="Hello", context=None)  # No context passed

        assert sent_messages[0].channel == "discord"
        assert sent_messages[0].chat_id == "channel-456"


class TestSpawnToolContext:
    """Tests for SpawnTool context support."""

    @pytest.mark.asyncio
    async def test_spawn_passes_context(self):
        """Test that SpawnTool passes context to subagent manager."""
        from nanobot.agent.tools.spawn import SpawnTool
        from nanobot.agent.subagent import SubagentManager
        from unittest.mock import MagicMock, AsyncMock

        manager = MagicMock(spec=SubagentManager)
        manager.spawn = AsyncMock(return_value="subagent-123")

        tool = SpawnTool(manager=manager)
        ctx = ToolContext(channel="telegram", chat_id="chat-123")

        result = await tool.execute(task="Hello", context=ctx)

        manager.spawn.assert_called_once()
        call_kwargs = manager.spawn.call_args.kwargs
        assert call_kwargs.get("origin_channel") == "telegram"
        assert call_kwargs.get("origin_chat_id") == "chat-123"

    @pytest.mark.asyncio
    async def test_spawn_fallback_to_global_state(self):
        """Test backward compatibility: fall back to global state if no context."""
        from nanobot.agent.tools.spawn import SpawnTool
        from nanobot.agent.subagent import SubagentManager
        from unittest.mock import MagicMock, AsyncMock

        manager = MagicMock(spec=SubagentManager)
        manager.spawn = AsyncMock(return_value="subagent-456")

        tool = SpawnTool(manager=manager)
        tool.set_context("discord", "channel-789")  # Set global state

        await tool.execute(task="Hello", context=None)  # No context passed

        manager.spawn.assert_called_once()
        call_kwargs = manager.spawn.call_args.kwargs
        assert call_kwargs.get("origin_channel") == "discord"
        assert call_kwargs.get("origin_chat_id") == "channel-789"


class TestCronToolContext:
    """Tests for CronTool context support."""

    @pytest.mark.asyncio
    async def test_cron_uses_context(self):
        """Test that CronTool uses context parameter for routing."""
        from nanobot.agent.tools.cron import CronTool
        from nanobot.cron.service import CronService
        from nanobot.cron.types import CronJob
        from unittest.mock import MagicMock

        # Create a mock job that will be returned by add_job
        mock_job = MagicMock(spec=CronJob)
        mock_job.id = "job-123"
        mock_job.name = "test reminder message"

        cron_service = MagicMock(spec=CronService)
        cron_service.add_job = MagicMock(return_value=mock_job)
        cron_service.list_jobs.return_value = []
        cron_service.remove_job.return_value = True

        tool = CronTool(cron_service)
        ctx = ToolContext(channel="telegram", chat_id="chat-123")

        result = await tool.execute(
            action="add", message="test reminder message", every_seconds=60, context=ctx
        )

        # Verify the job was added with correct channel/chat_id from context
        cron_service.add_job.assert_called_once()
        call_kwargs = cron_service.add_job.call_args.kwargs
        assert call_kwargs.get("channel") == "telegram"
        assert call_kwargs.get("to") == "chat-123"
        assert "job-123" in result

    @pytest.mark.asyncio
    async def test_cron_fallback_to_global_state(self):
        """Test backward compatibility: fall back to global state if no context."""
        from nanobot.agent.tools.cron import CronTool
        from nanobot.cron.service import CronService
        from nanobot.cron.types import CronJob
        from unittest.mock import MagicMock

        mock_job = MagicMock(spec=CronJob)
        mock_job.id = "job-456"
        mock_job.name = "fallback test"

        cron_service = MagicMock(spec=CronService)
        cron_service.add_job = MagicMock(return_value=mock_job)

        tool = CronTool(cron_service)
        tool.set_context("discord", "channel-789")  # Set global state

        await tool.execute(
            action="add", message="fallback test", every_seconds=120, context=None
        )

        # Verify global state was used
        cron_service.add_job.assert_called_once()
        call_kwargs = cron_service.add_job.call_args.kwargs
        assert call_kwargs.get("channel") == "discord"
        assert call_kwargs.get("to") == "channel-789"

    @pytest.mark.asyncio
    async def test_cron_no_context_error(self):
        """Test that cron returns error when no context is available."""
        from nanobot.agent.tools.cron import CronTool
        from nanobot.cron.service import CronService
        from unittest.mock import MagicMock

        cron_service = MagicMock(spec=CronService)

        tool = CronTool(cron_service)
        # Don't set context or pass it

        result = await tool.execute(
            action="add", message="test", every_seconds=60, context=None
        )

        assert "Error" in result
        assert "no session context" in result.lower()
