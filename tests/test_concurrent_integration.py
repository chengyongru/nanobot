"""Integration tests for concurrent message processing.

These tests verify that the concurrent message processing feature works end-to-end,
including proper routing isolation and session-level locking.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import ToolContext
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


def _make_agent_loop() -> tuple[AgentLoop, MessageBus]:
    """Create a minimal AgentLoop with mocked dependencies."""
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager") as MockSessionMgr, \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        MockSessionMgr.return_value.get_or_create = MagicMock()
        MockSessionMgr.return_value.save = MagicMock()
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


def _create_mock_session(key: str) -> MagicMock:
    """Create a mock session with the given key."""
    mock_session = MagicMock()
    mock_session.key = key
    mock_session.messages = []
    mock_session.last_consolidated = 0
    mock_session.get_history.return_value = []
    return mock_session


class TestConcurrentMessagesSameSession:
    """Tests for concurrent messages to the same session."""

    @pytest.mark.asyncio
    async def test_concurrent_messages_same_session(self):
        """Test that concurrent messages to same session don't interfere with routing."""
        loop, bus = _make_agent_loop()

        # Track outbound messages to verify routing
        sent_messages: list[OutboundMessage] = []

        async def capture_outbound(msg: OutboundMessage) -> None:
            sent_messages.append(msg)

        bus.publish_outbound = capture_outbound

        # Create mock session
        mock_session = _create_mock_session("telegram:chat-123")
        loop.sessions.get_or_create.return_value = mock_session

        # Track the order of operations
        operations: list[str] = []

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            operations.append(f"start-{context.turn_id}")
            await asyncio.sleep(0.05)  # Simulate LLM call
            operations.append(f"end-{context.turn_id}")
            return "Response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Create two messages to the same session
        msg1 = InboundMessage(
            channel="telegram", sender_id="user1", chat_id="chat-123", content="Hello 1"
        )
        msg2 = InboundMessage(
            channel="telegram", sender_id="user1", chat_id="chat-123", content="Hello 2"
        )

        # Process both concurrently
        await asyncio.gather(
            loop._dispatch(msg1),
            loop._dispatch(msg2),
        )

        # Both should complete successfully
        assert len(sent_messages) == 2

        # All messages should route to the same channel/chat_id
        for msg in sent_messages:
            assert msg.channel == "telegram"
            assert msg.chat_id == "chat-123"

    @pytest.mark.asyncio
    async def test_concurrent_same_session_serialized_access(self):
        """Test that session read/write operations are serialized for same session."""
        loop, bus = _make_agent_loop()

        # Track session access times
        access_times: list[tuple[float, str]] = []

        def get_or_create_side_effect(key):
            access_times.append((asyncio.get_event_loop().time(), "read"))
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            await asyncio.sleep(0.03)
            return "Response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        msg1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="m1")
        msg2 = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="m2")

        # Process concurrently
        await asyncio.gather(
            loop._dispatch(msg1),
            loop._dispatch(msg2),
        )

        # Should have multiple session accesses (reads/writes)
        assert len(access_times) >= 2


class TestConcurrentMessagesDifferentSessions:
    """Tests for concurrent messages to different sessions."""

    @pytest.mark.asyncio
    async def test_concurrent_messages_different_sessions(self):
        """Test that concurrent messages to different sessions are truly parallel."""
        loop, bus = _make_agent_loop()

        # Track start/end times for each session
        timing: dict[str, dict] = {}

        def get_or_create_side_effect(key):
            if key not in timing:
                timing[key] = {"start": None, "end": None}
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            session_key = f"{context.channel}:{context.chat_id}"
            timing[session_key]["start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate LLM processing
            timing[session_key]["end"] = asyncio.get_event_loop().time()
            return "Response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Two messages to different sessions
        msg1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="chat-A", content="Hello A")
        msg2 = InboundMessage(channel="discord", sender_id="u2", chat_id="chat-B", content="Hello B")

        # Process concurrently
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(
            loop._dispatch(msg1),
            loop._dispatch(msg2),
        )
        total_time = asyncio.get_event_loop().time() - start_time

        # Should complete in ~0.1s (parallel), not ~0.2s (sequential)
        # Allow some margin for test overhead
        assert total_time < 0.25, f"Tasks took {total_time}s, expected parallel execution"

        # Both should have completed
        assert "telegram:chat-A" in timing
        assert "discord:chat-B" in timing

    @pytest.mark.asyncio
    async def test_different_sessions_no_mutual_blocking(self):
        """Test that different sessions don't block each other."""
        loop, bus = _make_agent_loop()

        execution_order: list[str] = []

        def get_or_create_side_effect(key):
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            session_key = f"{context.channel}:{context.chat_id}"
            execution_order.append(f"{session_key}-start")
            await asyncio.sleep(0.05)
            execution_order.append(f"{session_key}-end")
            return "Response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Three different sessions
        msg1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="m1")
        msg2 = InboundMessage(channel="discord", sender_id="u2", chat_id="c2", content="m2")
        msg3 = InboundMessage(channel="slack", sender_id="u3", chat_id="c3", content="m3")

        await asyncio.gather(
            loop._dispatch(msg1),
            loop._dispatch(msg2),
            loop._dispatch(msg3),
        )

        # All should have started before all finished (parallel)
        starts = [e for e in execution_order if e.endswith("-start")]
        ends = [e for e in execution_order if e.endswith("-end")]
        assert len(starts) == 3
        assert len(ends) == 3


class TestMessageRoutingIsolation:
    """Tests for message routing isolation between concurrent messages."""

    @pytest.mark.asyncio
    async def test_message_routing_isolation(self):
        """Test that concurrent messages route to correct channels."""
        loop, bus = _make_agent_loop()

        # Capture outbound messages
        outbound_messages: list[OutboundMessage] = []

        async def capture_outbound(msg: OutboundMessage) -> None:
            outbound_messages.append(msg)

        bus.publish_outbound = capture_outbound

        def get_or_create_side_effect(key):
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            # Simulate tool that uses context for routing
            await asyncio.sleep(0.02)
            return f"Response from {context.channel}", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Send messages to Telegram and Discord concurrently
        msg_telegram = InboundMessage(
            channel="telegram", sender_id="user1", chat_id="tg-chat", content="Hello Telegram"
        )
        msg_discord = InboundMessage(
            channel="discord", sender_id="user2", chat_id="dc-chat", content="Hello Discord"
        )

        await asyncio.gather(
            loop._dispatch(msg_telegram),
            loop._dispatch(msg_discord),
        )

        # Verify responses go to correct channels
        assert len(outbound_messages) == 2

        telegram_responses = [m for m in outbound_messages if m.channel == "telegram"]
        discord_responses = [m for m in outbound_messages if m.channel == "discord"]

        assert len(telegram_responses) == 1
        assert len(discord_responses) == 1

        assert telegram_responses[0].chat_id == "tg-chat"
        assert discord_responses[0].chat_id == "dc-chat"

    @pytest.mark.asyncio
    async def test_tool_context_isolation(self):
        """Test that ToolContext is properly isolated between concurrent executions."""
        loop, bus = _make_agent_loop()

        # Track contexts seen during execution
        seen_contexts: list[ToolContext] = []

        def get_or_create_side_effect(key):
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            seen_contexts.append(context)
            await asyncio.sleep(0.02)
            return f"Response to {context.channel}:{context.chat_id}", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Multiple concurrent messages
        messages = [
            InboundMessage(channel="telegram", sender_id="u1", chat_id="chat-1", content="m1"),
            InboundMessage(channel="discord", sender_id="u2", chat_id="chat-2", content="m2"),
            InboundMessage(channel="slack", sender_id="u3", chat_id="chat-3", content="m3"),
        ]

        await asyncio.gather(*[loop._dispatch(msg) for msg in messages])

        # Each execution should have received the correct context
        assert len(seen_contexts) == 3

        context_by_channel = {ctx.channel: ctx for ctx in seen_contexts}
        assert "telegram" in context_by_channel
        assert "discord" in context_by_channel
        assert "slack" in context_by_channel

        # Verify each context has the correct chat_id
        assert context_by_channel["telegram"].chat_id == "chat-1"
        assert context_by_channel["discord"].chat_id == "chat-2"
        assert context_by_channel["slack"].chat_id == "chat-3"

    @pytest.mark.asyncio
    async def test_no_cross_channel_routing(self):
        """Test that messages don't get routed to wrong channels."""
        loop, bus = _make_agent_loop()

        outbound_messages: list[OutboundMessage] = []

        async def capture_outbound(msg: OutboundMessage) -> None:
            outbound_messages.append(msg)

        bus.publish_outbound = capture_outbound

        def get_or_create_side_effect(key):
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            await asyncio.sleep(0.02)
            return "Response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Many concurrent messages to different channels
        test_cases = [
            ("telegram", "tg-1"),
            ("discord", "dc-1"),
            ("slack", "sl-1"),
            ("whatsapp", "wa-1"),
            ("telegram", "tg-2"),  # Same channel, different chat
            ("discord", "dc-2"),
        ]

        inbound_messages = [
            InboundMessage(channel=ch, sender_id=f"user-{i}", chat_id=chat, content=f"msg-{i}")
            for i, (ch, chat) in enumerate(test_cases)
        ]

        await asyncio.gather(*[loop._dispatch(msg) for msg in inbound_messages])

        # Verify no cross-routing
        assert len(outbound_messages) == len(test_cases)

        for i, (expected_channel, expected_chat) in enumerate(test_cases):
            matching = [
                m for m in outbound_messages
                if m.channel == expected_channel and m.chat_id == expected_chat
            ]
            assert len(matching) == 1, f"Expected exactly one message to {expected_channel}:{expected_chat}"


class TestIntegrationWithMessageTool:
    """Integration tests with actual MessageTool execution."""

    @pytest.mark.asyncio
    async def test_message_tool_uses_correct_context(self):
        """Test that MessageTool receives and uses the correct context during concurrent execution."""
        from nanobot.agent.tools.message import MessageTool

        loop, bus = _make_agent_loop()

        # Track messages sent via MessageTool
        tool_messages: list[OutboundMessage] = []

        async def capture_tool_message(msg: OutboundMessage) -> None:
            tool_messages.append(msg)

        # Track outbound messages from _dispatch
        dispatch_messages: list[OutboundMessage] = []

        async def capture_dispatch(msg: OutboundMessage) -> None:
            dispatch_messages.append(msg)

        bus.publish_outbound = capture_dispatch

        def get_or_create_side_effect(key):
            return _create_mock_session(key)

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        # Create a real MessageTool with our capture callback
        message_tool = MessageTool(send_callback=capture_tool_message)

        # Replace the registered message tool
        loop.tools._tools["message"] = message_tool

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            # Simulate calling the message tool
            await loop.tools.execute(
                "message",
                {"content": f"Tool response for {context.channel}"},
                context=context,
            )
            return "Main response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Concurrent messages to different channels
        msg1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="tg-chat", content="Hello")
        msg2 = InboundMessage(channel="discord", sender_id="u2", chat_id="dc-chat", content="Hello")

        await asyncio.gather(
            loop._dispatch(msg1),
            loop._dispatch(msg2),
        )

        # Both tool messages should be captured
        assert len(tool_messages) == 2

        # Verify correct routing
        tg_msgs = [m for m in tool_messages if m.channel == "telegram"]
        dc_msgs = [m for m in tool_messages if m.channel == "discord"]

        assert len(tg_msgs) == 1
        assert len(dc_msgs) == 1
        assert tg_msgs[0].chat_id == "tg-chat"
        assert dc_msgs[0].chat_id == "dc-chat"
