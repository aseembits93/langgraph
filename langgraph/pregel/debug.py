from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph._internal._config import patch_checkpoint_map
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_NS,
    ERROR,
    INTERRUPT,
    NS_END,
    NS_SEP,
    RETURN,
)
from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import CheckpointMetadata, PendingWrite
from langgraph.constants import TAG_HIDDEN
from langgraph.pregel._io import read_channels
from langgraph.types import PregelExecutableTask, PregelTask, StateSnapshot

_BOLD_PREFIX = "\033[1m"

_BOLD_SUFFIX = "\033[0m"

__all__ = ("TaskPayload", "TaskResultPayload", "CheckpointTask", "CheckpointPayload")


class TaskPayload(TypedDict):
    id: str
    name: str
    input: Any
    triggers: list[str]


class TaskResultPayload(TypedDict):
    id: str
    name: str
    error: str | None
    interrupts: list[dict]
    result: list[tuple[str, Any]]


class CheckpointTask(TypedDict):
    id: str
    name: str
    error: str | None
    interrupts: list[dict]
    state: RunnableConfig | None


class CheckpointPayload(TypedDict):
    config: RunnableConfig | None
    metadata: CheckpointMetadata
    values: dict[str, Any]
    next: list[str]
    parent_config: RunnableConfig | None
    tasks: list[CheckpointTask]


TASK_NAMESPACE = UUID("6ba7b831-9dad-11d1-80b4-00c04fd430c8")


def map_debug_tasks(tasks: Iterable[PregelExecutableTask]) -> Iterator[TaskPayload]:
    """Produce "task" events for stream_mode=debug."""
    for task in tasks:
        if task.config is not None and TAG_HIDDEN in task.config.get("tags", []):
            continue

        yield {
            "id": task.id,
            "name": task.name,
            "input": task.input,
            "triggers": task.triggers,
        }


def map_debug_task_results(
    task_tup: tuple[PregelExecutableTask, Sequence[tuple[str, Any]]],
    stream_keys: str | Sequence[str],
) -> Iterator[TaskResultPayload]:
    """Produce "task_result" events for stream_mode=debug."""
    stream_channels_list = (
        [stream_keys] if isinstance(stream_keys, str) else stream_keys
    )
    task, writes = task_tup
    yield {
        "id": task.id,
        "name": task.name,
        "error": next((w[1] for w in writes if w[0] == ERROR), None),
        "result": [w for w in writes if w[0] in stream_channels_list or w[0] == RETURN],
        "interrupts": [
            asdict(v)
            for w in writes
            if w[0] == INTERRUPT
            for v in (w[1] if isinstance(w[1], Sequence) else [w[1]])
        ],
    }


def rm_pregel_keys(config: RunnableConfig | None) -> RunnableConfig | None:
    """Remove pregel-specific keys from the config."""
    if config is None:
        return config
    return {
        "configurable": {
            k: v
            for k, v in config.get("configurable", {}).items()
            if not k.startswith("__pregel_")
        }
    }


def map_debug_checkpoint(
    config: RunnableConfig,
    channels: Mapping[str, BaseChannel],
    stream_channels: str | Sequence[str],
    metadata: CheckpointMetadata,
    tasks: Iterable[PregelExecutableTask],
    pending_writes: list[PendingWrite],
    parent_config: RunnableConfig | None,
    output_keys: str | Sequence[str],
) -> Iterator[CheckpointPayload]:
    """Produce "checkpoint" events for stream_mode=debug."""

    parent_ns = config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
    task_states: dict[str, RunnableConfig | StateSnapshot] = {}

    for task in tasks:
        if not task.subgraphs:
            continue

        # assemble checkpoint_ns for this task
        task_ns = f"{task.name}{NS_END}{task.id}"
        if parent_ns:
            task_ns = f"{parent_ns}{NS_SEP}{task_ns}"

        # set config as signal that subgraph checkpoints exist
        task_states[task.id] = {
            CONF: {
                "thread_id": config[CONF]["thread_id"],
                CONFIG_KEY_CHECKPOINT_NS: task_ns,
            }
        }

    yield {
        "config": rm_pregel_keys(patch_checkpoint_map(config, metadata)),
        "parent_config": rm_pregel_keys(patch_checkpoint_map(parent_config, metadata)),
        "values": read_channels(channels, stream_channels),
        "metadata": metadata,
        "next": [t.name for t in tasks],
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "error": t.error,
                "state": t.state,
            }
            if t.error
            else {
                "id": t.id,
                "name": t.name,
                "result": t.result,
                "interrupts": tuple(asdict(i) for i in t.interrupts),
                "state": t.state,
            }
            if t.result
            else {
                "id": t.id,
                "name": t.name,
                "interrupts": tuple(asdict(i) for i in t.interrupts),
                "state": t.state,
            }
            for t in tasks_w_writes(tasks, pending_writes, task_states, output_keys)
        ],
    }


def tasks_w_writes(
    tasks: Iterable[PregelTask | PregelExecutableTask],
    pending_writes: list[PendingWrite] | None,
    states: dict[str, RunnableConfig | StateSnapshot] | None,
    output_keys: str | Sequence[str],
) -> tuple[PregelTask, ...]:
    """Apply writes / subgraph states to tasks to be returned in a StateSnapshot."""
    pending_writes = pending_writes or []

    # Pre-group/write index: id -> {chan: [val, ...]}
    id2chan2values: dict[str, dict[str, list]] = {}
    if pending_writes:
        for tid, chan, val in pending_writes:
            taskdict = id2chan2values.setdefault(tid, {})
            taskdict.setdefault(chan, []).append(val)

    # Pretest: str or sequence for fast path in output_keys logic
    is_str_output_keys = isinstance(output_keys, str)
    output_keys_set = set(output_keys) if not is_str_output_keys else None

    out: list[PregelTask] = []
    for task in tasks:
        t_id = task.id
        writes = id2chan2values.get(t_id, {})

        # RETURN
        rtn = writes.get(RETURN, [MISSING])[0] if writes.get(RETURN) else MISSING

        # ERROR
        error = writes.get(ERROR, [None])[0] if writes.get(ERROR) else None

        # INTERRUPT - always tuple, flattening if any value is a Sequence except str/bytes
        ivals = []
        for v in writes.get(INTERRUPT, []):
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                ivals.extend(v)
            else:
                ivals.append(v)
        interrupts = tuple(ivals)

        # State lookup
        state_val = states.get(t_id) if states else None

        # Output
        out_field = None
        # Only bother if we have at least 1 write for this id that's not ERROR/INTERRUPT.
        has_any_non_exclusive = False
        for chan in writes:
            if chan not in (ERROR, INTERRUPT):
                has_any_non_exclusive = True
                break
        if has_any_non_exclusive:
            if rtn is not MISSING:
                out_field = rtn
            elif is_str_output_keys:
                # Look for first write with chan == output_keys
                res = writes.get(output_keys)
                if res:
                    out_field = res[0]
                else:
                    out_field = None
            else:
                # Sequence case: gather all matching output_keys as dict
                result = {}
                for chan in writes:
                    if chan in output_keys_set:
                        # keep "last" value if channel appears multiple times
                        result[chan] = writes[chan][-1]
                out_field = result

        out.append(PregelTask(
            t_id,
            task.name,
            task.path,
            error,
            interrupts,
            state_val,
            out_field,
        ))

    return tuple(out)


COLOR_MAPPING = {
    "black": "0;30",
    "red": "0;31",
    "green": "0;32",
    "yellow": "0;33",
    "blue": "0;34",
    "magenta": "0;35",
    "cyan": "0;36",
    "white": "0;37",
    "gray": "1;30",
}


def get_colored_text(text: str, color: str) -> str:
    """Get colored text."""
    return f"\033[1;3{COLOR_MAPPING[color]}m{text}\033[0m"


def get_bolded_text(text: str) -> str:
    """Get bolded text."""
    # Precompute constant parts to avoid repeated string parsing at runtime
    return "\033[1m" + text + "\033[0m"
