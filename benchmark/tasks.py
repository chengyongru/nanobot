"""Benchmark framework for measuring plan tool effectiveness.

Design principle: measure whether planning *actually helps* the agent
complete tasks better, not just whether the agent *uses* the plan tool.

Metrics:
  - task_completion: 0.0-1.0, whether the task was fully completed
  - correctness: 0.0-1.0, whether the result is correct
  - tool_call_efficiency: fewer tool calls = better (less waste)
  - token_efficiency: fewer tokens = better (less overhead)
  - error_recovery: whether the agent recovers from mid-task errors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""
    name: str
    prompt: str
    category: str  # "code_edit", "debug", "multi_file", "research"
    complexity: str  # "simple", "medium", "complex"
    validation: str  # Description of what a correct answer looks like
    expected_steps: int = 0  # Rough expected number of tool calls


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    task_name: str
    with_plan: bool
    completion_score: float  # 0.0-1.0
    correctness_score: float  # 0.0-1.0
    tool_calls: int
    total_tokens: int = 0
    plan_used: bool = False
    duration_seconds: float = 0.0
    error_recovery: bool = False
    raw_output: str = ""
    notes: str = ""


@dataclass
class BenchmarkSuite:
    """A collection of benchmark results."""
    name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        with_plan = [r for r in self.results if r.with_plan]
        without_plan = [r for r in self.results if not r.with_plan]

        def avg(items, key):
            vals = [getattr(r, key) for r in items]
            return sum(vals) / len(vals) if vals else 0

        return {
            "name": self.name,
            "tasks": len(self.results),
            "with_plan": {
                "count": len(with_plan),
                "avg_completion": avg(with_plan, "completion_score"),
                "avg_correctness": avg(with_plan, "correctness_score"),
                "avg_tool_calls": avg(with_plan, "tool_calls"),
                "plan_usage_rate": sum(1 for r in with_plan if r.plan_used) / max(len(with_plan), 1),
            },
            "without_plan": {
                "count": len(without_plan),
                "avg_completion": avg(without_plan, "completion_score"),
                "avg_correctness": avg(without_plan, "correctness_score"),
                "avg_tool_calls": avg(without_plan, "tool_calls"),
            },
            "delta_completion": avg(with_plan, "completion_score") - avg(without_plan, "completion_score"),
            "delta_correctness": avg(with_plan, "correctness_score") - avg(without_plan, "correctness_score"),
        }


# Predefined benchmark tasks
BENCHMARK_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        name="add_logging",
        prompt=(
            "Add structured logging to the function `process_data(data: dict) -> dict` "
            "in `nanobot/utils/helpers.py`. The logging should: 1) Log when processing starts "
            "with the input keys, 2) Log when processing completes with the output keys, "
            "3) Log any errors with the full traceback. Use the existing loguru logger."
        ),
        category="code_edit",
        complexity="medium",
        validation="Function has logging at start, end, and error points using loguru",
        expected_steps=5,
    ),
    BenchmarkTask(
        name="refactor_config",
        prompt=(
            "The function `load_config()` in `nanobot/config/loader.py` is too long. "
            "Refactor it by: 1) Extracting the validation logic into a separate function "
            "`_validate_config(config: dict) -> list[str]`, 2) Extracting the default "
            "merging logic into `_merge_defaults(config: dict) -> dict`. "
            "Keep the public API unchanged."
        ),
        category="multi_file",
        complexity="complex",
        validation="load_config is shorter, two new private functions exist, public API unchanged",
        expected_steps=8,
    ),
    BenchmarkTask(
        name="fix_off_by_one",
        prompt=(
            "There's a bug in the pagination logic of `GrepTool` in "
            "`nanobot/agent/tools/search.py`. When `page=2` is requested, "
            "it shows the same results as page 1. Find and fix the bug."
        ),
        category="debug",
        complexity="medium",
        validation="Pagination correctly offsets results for page > 1",
        expected_steps=6,
    ),
    BenchmarkTask(
        name="add_channel_health",
        prompt=(
            "Add a health check endpoint to the channel system. "
            "Create a new file `nanobot/channels/health.py` that: "
            "1) Defines `ChannelHealth` dataclass with status, latency_ms, last_error fields, "
            "2) Implements `check_channel_health(channel_name: str) -> ChannelHealth`, "
            "3) Integrates with the existing ChannelManager. "
            "Do NOT modify any existing files except to register the new module."
        ),
        category="multi_file",
        complexity="complex",
        validation="New file exists, health check works, minimal changes to existing files",
        expected_steps=10,
    ),
    BenchmarkTask(
        name="optimize_imports",
        prompt=(
            "The module `nanobot/agent/runner.py` has several unused imports. "
            "Find and remove them. Also, if there are any imports that could be "
            "moved inside functions (to avoid circular imports or slow startup), "
            "move them."
        ),
        category="code_edit",
        complexity="simple",
        validation="Unused imports removed, necessary imports kept, no runtime errors",
        expected_steps=3,
    ),
    BenchmarkTask(
        name="error_handling_chain",
        prompt=(
            "In `nanobot/agent/tools/filesystem.py`, the `ReadFileTool` handles "
            "permission errors by returning a generic error string. Improve this by: "
            "1) Returning specific error messages for common cases (permission denied, "
            "file not found, encoding error, binary file), 2) Adding a suggestion "
            "for what the user might want to do instead."
        ),
        category="code_edit",
        complexity="medium",
        validation="Specific error messages for at least 3 error types with suggestions",
        expected_steps=5,
    ),
    BenchmarkTask(
        name="implement_tool_timeout",
        prompt=(
            "Add a configurable timeout to tool execution. Currently tools can run "
            "indefinitely. Add: 1) A `timeout_seconds` field to the tool config schema, "
            "2) Execution wrapping with asyncio.wait_for in the runner, "
            "3) A friendly error message when timeout occurs."
        ),
        category="multi_file",
        complexity="complex",
        validation="Timeout works, config schema has the field, runner respects it",
        expected_steps=10,
    ),
    BenchmarkTask(
        name="sort_tool_list",
        prompt=(
            "List all Python files in the `nanobot/agent/tools/` directory, "
            "show their line counts, and tell me which tool has the most code."
        ),
        category="research",
        complexity="simple",
        validation="All tool files listed with correct line counts, largest identified",
        expected_steps=3,
    ),
]
