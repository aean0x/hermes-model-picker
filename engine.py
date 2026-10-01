"""model-picker — handoff context engine.

A transparent subclass of Hermes's built-in ``ContextCompressor`` that adds two
capabilities:

1. Request-scoped context replacement for mid-turn escalation. When the
   ``escalate_model`` tool fires, it stashes a structured handoff on the router;
   ``select_context`` then swaps the outgoing request for a compact handoff
   (system prompt + summary + verbatim tail).
2. One-shot forced compaction on a classifier-driven tier change. The router
   marks the session; ``should_compress_info`` returns True once, so the
   built-in ``compress()`` runs at the next turn boundary before the incoming
   model cold-reads the transcript.

Persisted history is mutated only by the built-in compaction path; the
escalation handoff is request-only, so prompt cache and the conversation DB
stay coherent.

The router state lives in the plugin (its hooks and tool own the session
lifecycle), so the engine reads it through the small ``bind_router`` seam
instead of reaching into the plugin module. The plugin system holds one shared
engine instance and hands each agent a deepcopy (``clone_for_agent``), which is
why the tail budget is instance state and the pending handoff/compaction are
keyed by session, not stored on the instance.

Every other method is inherited unchanged from ``ContextCompressor``, so normal
compaction (``should_compress`` / ``compress`` / ``update_model`` /
``model_thresholds``) is byte-for-byte the built-in behavior.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.context_compressor import ContextCompressor


def _load_settings() -> Any:
    try:
        from . import settings as settings_mod

        return settings_mod
    except ImportError:
        path = Path(__file__).resolve().parent / "settings.py"
        spec = importlib.util.spec_from_file_location("_model_picker_settings", path)
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_settings = _load_settings()

ENGINE_NAME = _settings.ENGINE_NAME
_DEFAULT_TAIL_CHARS = _settings.HANDOFF_TAIL_CHARS

# Injected by the plugin at register() time (see the module docstring).
_TAKE_HANDOFF: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
_TAKE_COMPACT: Optional[Callable[[str], bool]] = None


def bind_router(
    *,
    take_handoff: Callable[[str], Optional[Dict[str, Any]]],
    take_compact: Callable[[str], bool],
) -> None:
    """Wire the engine to the router's session-keyed handoff/compaction state."""
    global _TAKE_HANDOFF, _TAKE_COMPACT
    _TAKE_HANDOFF = take_handoff
    _TAKE_COMPACT = take_compact


def _format_handoff(handoff: Dict[str, Any]) -> str:
    """Render the structured handoff the escalating model supplied."""
    dest = (handoff.get("to_tier") or "high").strip()
    model = (handoff.get("to_model") or "").strip()
    src = (handoff.get("from_tier") or "").strip()
    dest_label = f"{dest}" + (f" ({model})" if model else "")
    src_label = f" from {src}" if src else ""
    lines = [
        f"CONTEXT HANDOFF (model escalation{src_label} → {dest_label}): "
        "the previous model summarised the conversation and its in-flight "
        "task so you can continue without re-reading the full history. "
        "Continue from here.",
        "",
    ]
    for key, label in (
        ("summary", "Conversation summary"),
        ("task_state", "Current task and intent"),
        ("tried_so_far", "Already tried"),
        ("failure_point", "Where it failed"),
        ("next_hypothesis", "Next hypothesis"),
    ):
        value = (handoff.get(key) or "").strip()
        if value:
            lines.append(f"## {label}")
            lines.append(value)
            lines.append("")
    return "\n".join(lines).strip()


def _text_of(message: Dict[str, Any]) -> str:
    """Flatten a message's content to plain text for budget accounting."""
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            (part.get("text", "") if isinstance(part, dict) else str(part))
            for part in content
        )
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _starts_at_turn_boundary(tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a leading partial tool group so the tail opens on a user message.

    An assistant message carrying ``tool_calls`` must be followed by the tool
    results that answer it; a tail that starts mid-group is a malformed request
    for every provider. Trim forward to the first user message instead.
    """
    for index, message in enumerate(tail):
        if message.get("role") == "user":
            return tail[index:]
    return tail


class ModelPickerContextEngine(ContextCompressor):
    """Built-in compressor plus a request-scoped escalation handoff."""

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self, *args: Any, model: str = "", name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, model=model, **kwargs)
        self._name = name or ENGINE_NAME
        self._sid: str = ""
        self._handoff_tail_chars: int = int(_DEFAULT_TAIL_CHARS)

    def clone_for_agent(self) -> "ModelPickerContextEngine":
        """Per-agent clone; module-level router state is shared, not copied."""
        return copy.deepcopy(self)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """Remember which session this instance serves (router state is keyed by it)."""
        self._sid = session_id or ""
        super().on_session_start(session_id, **kwargs)

    def should_compress_info(
        self, prompt_tokens: int | None = None
    ) -> tuple[bool, str | None]:
        """Honor a one-shot router compaction request, else defer to base.

        A classifier-driven model switch invalidates the incoming model's prompt
        cache anyway, so this is the safe moment to compact the transcript into
        a summary before the new model reads it. Returning ``(True, None)``
        bypasses the base cooldown/anti-thrash guards exactly once; every later
        call falls through to the inherited threshold logic.
        """
        take = _TAKE_COMPACT
        if self._sid and take is not None and take(self._sid):
            return True, None
        return super().should_compress_info(prompt_tokens)

    def _tail(self, source: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recent verbatim tail, whole messages, charged against the tail budget.

        Messages are carried through unchanged, so ``tool_calls``,
        ``tool_call_id`` and ``name`` survive the handoff — a rebuilt
        ``{role, content}`` pair silently orphaned every tool group in the tail.
        """
        tail: List[Dict[str, Any]] = []
        budget = self._handoff_tail_chars
        for message in reversed(source or []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                continue
            text = _text_of(message).strip()
            if not text and not message.get("tool_calls"):
                continue
            tail.insert(0, message)
            budget -= len(text)
            if budget <= 0:
                break
        return _starts_at_turn_boundary(tail)

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Replace the request with a compact handoff when an escalation is pending.

        Returns ``None`` (no-op → byte-identical request, cache preserved) when
        no handoff is pending for this session.
        """
        take = _TAKE_HANDOFF
        handoff = take(self._sid) if (self._sid and take is not None) else None
        if not handoff:
            return None
        try:
            out: list[Dict[str, Any]] = []

            # Preserve the system prompt verbatim — it carries the stable
            # prefix and the (already-switched) Model:/Provider: footer.
            system = request_messages[0] if request_messages else None
            if isinstance(system, dict) and system.get("role") == "system":
                out.append(system)

            out.append({"role": "user", "content": _format_handoff(handoff)})

            # Prefer the live request (includes this turn's tool loop / failure
            # text) over persisted history.
            out.extend(self._tail(request_messages or conversation_messages or []))
            return out
        except Exception:
            # Fail-open: if handoff construction breaks, fall back to the
            # unmodified request rather than breaking the turn.
            return None
