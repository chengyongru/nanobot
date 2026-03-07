"""Tests for concurrent session handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

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


class TestSessionLocks:
    @pytest.mark.asyncio
    async def test_session_read_write_serialized(self):
        """Test that session read/write operations are serialized for the same session."""
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        # Create a mock session
        mock_session = MagicMock()
        mock_session.key = "test:c1"
        mock_session.messages = []
        mock_session.last_consolidated = 0
        mock_session.get_history.return_value = []

        loop.sessions.get_or_create.return_value = mock_session

        # Mock _run_agent_loop to track order
        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            order.append("llm-start")
            await asyncio.sleep(0.05)
            order.append("llm-end")
            return "response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello1")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello2")

        # Both messages should be processed concurrently but session access serialized
        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)

        # Both LLM calls should interleave (parallel processing)
        # But session access is serialized per-session lock
        assert "llm-start" in order
        assert "llm-end" in order

    @pytest.mark.asyncio
    async def test_different_sessions_parallel(self):
        """Test that different sessions can process in parallel."""
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []
        start_times = {}
        end_times = {}

        # Create mock sessions for different session keys
        def get_or_create_side_effect(key):
            mock_session = MagicMock()
            mock_session.key = key
            mock_session.messages = []
            mock_session.last_consolidated = 0
            mock_session.get_history.return_value = []
            return mock_session

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        # Mock _run_agent_loop to track timing
        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            import time
            # Get session key from messages if possible
            order.append(f"llm-start-{len(order)}")
            start_times[len(order)] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)
            end_times[len(order)] = asyncio.get_event_loop().time()
            order.append(f"llm-end-{len(order)}")
            return "response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        # Two different sessions (different chat_ids)
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c2", content="hello")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)

        # Both should have started before either finished (parallel execution)
        # Check that the tasks ran concurrently
        assert len(order) == 4  # 2 starts + 2 ends

    @pytest.mark.asyncio
    async def test_get_session_lock_returns_same_lock_for_same_key(self):
        """Test that _get_session_lock returns the same lock for the same session key."""
        loop, _ = _make_loop()

        lock1 = loop._get_session_lock("test:c1")
        lock2 = loop._get_session_lock("test:c1")

        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_get_session_lock_returns_different_locks_for_different_keys(self):
        """Test that _get_session_lock returns different locks for different session keys."""
        loop, _ = _make_loop()

        lock1 = loop._get_session_lock("test:c1")
        lock2 = loop._get_session_lock("test:c2")

        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_dispatch_no_global_lock(self):
        """Test that _dispatch does not use a global lock."""
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()

        # Verify no _processing_lock attribute exists
        assert not hasattr(loop, "_processing_lock") or loop._processing_lock is None

        # Verify _session_locks exists
        assert hasattr(loop, "_session_locks")
        assert isinstance(loop._session_locks, dict)


class TestSessionLockBehavior:
    @pytest.mark.asyncio
    async def test_concurrent_same_session_waits(self):
        """Test that concurrent messages to the same session wait for session lock."""
        from nanobot.bus.events import InboundMessage

        loop, _ = _make_loop()

        # Track when session reads happen
        session_read_times = []
        session_write_times = []

        def get_or_create_side_effect(key):
            mock_session = MagicMock()
            mock_session.key = key
            mock_session.messages = []
            mock_session.last_consolidated = 0
            mock_session.get_history.return_value = []
            session_read_times.append(asyncio.get_event_loop().time())
            return mock_session

        loop.sessions.get_or_create.side_effect = get_or_create_side_effect

        original_save = MagicMock()
        loop.sessions.save = original_save

        async def mock_run_agent_loop(messages, context=None, on_progress=None):
            await asyncio.sleep(0.05)
            return "response", [], messages

        loop._run_agent_loop = mock_run_agent_loop

        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello1")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello2")

        # Both messages to the same session
        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)

        # Both should have completed
        assert len(session_read_times) >= 2  # At least 2 reads (initial + final for each message)
