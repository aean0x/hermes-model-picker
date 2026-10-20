"""model-picker — classify each turn, pick a model tier, on public surfaces only.

Exactly three models: low < default < high. Models, providers, labels, and
escalation are config — see settings.py and config.default.json. Override via a
plugin-adjacent config.json, a MODEL_PICKER_CONFIG path, or MODEL_PICKER_* env
vars. "medium" is the deprecated alias for "default" and still pins the default
slot.

Every extension point here is a documented plugin surface:

  • rating      — ``pre_llm_call`` observes the turn and produces the tier;
  • model choice — ``llm_request`` middleware rewrites the outgoing request's
    ``model`` before the provider call. Nothing else is touched: no client
    rebuild, no ``AIAgent`` attribute writes, no method wrapping, and no WebUI
    DOM script;
  • classifier  — ``ctx.llm`` (host-owned completions on the active model);
  • handoff     — ``ctx.register_context_engine`` (the public single-engine
    slot), active only when the host config names this engine.

Policy:
  • Each real user turn is classified once; the work loop stays on that tier
    for the whole multi-tool turn.
  • Classifier is 3-way (low/default/high). High is rare (money /
    irreversible / security). Prefer low on doubt.
  • The classifier is told the previous tier and biased to keep it when the
    scope/topic is unchanged (low/default only; high is never sticky).
  • A classifier-driven tier change compacts the transcript before the switch
    so the new model cold-reads a summary (explicit pins/escalation skip this).
  • Consecutive tool errors stage an escalation checkpoint; the working model
    calls escalate_model to climb one tier.
  • Manual /low /default /high pins pause auto-routing until /auto. /medium is
    the deprecated form of /default.
  • Slash pins must start the message; a mid-paragraph /high is not a pin.

Tier scope. An ``llm_request`` middleware can rewrite the model id, not the
API host: the client for the request is built from the session's provider. All
three tiers must therefore resolve to the session's own provider. A tier that
names a different provider is refused (logged once) and the session's model
stands.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("plugins.model-picker")


def _import_settings() -> Any:
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


_s = _import_settings()
MODELS = _s.MODELS
NAMES = _s.NAMES
RANK = _s.RANK
LEGACY_NAMES = _s.LEGACY_NAMES
as_name = _s.as_name
_ESCALATE_MAX = _s.ESCALATE_MAX
_ESCALATION_ERRORS = _s.ESCALATION_ERRORS
_SKIP_PLATFORMS = _s.SKIP_PLATFORMS
_CLASSIFIER = _s.CLASSIFIER
_CLASSIFIER_CONTEXT_CHARS = _s.CLASSIFIER_CONTEXT_CHARS
_CLASSIFIER_TIMEOUT_S = _s.CLASSIFIER_TIMEOUT_S
_CLASSIFIER_MODEL = getattr(_s, "CLASSIFIER_MODEL", "")
_CLASSIFIER_PROVIDER = getattr(_s, "CLASSIFIER_PROVIDER", "")
# False when no tier has a model id and the host has no primary model either.
# The full surface still registers (declared == registered for the validator);
# the hooks simply decline to classify, so registration never raises.
_CONFIGURED = bool(getattr(_s, "CONFIGURED", True))
_UNCONFIGURED_SLOTS = tuple(getattr(_s, "UNCONFIGURED_SLOTS", ()))
_MIN = "low"
_MID = "default"
_TOP = NAMES[-1]  # "high"

# Host-owned PluginContext (set in register()). Only public attributes are read
# from it: .llm for the classifier, .get_config for operator knobs.
_ctx: Any = None
# The single engine instance the plugin system holds. Each agent gets a clone.
_engine: Any = None


def _attach_file_handler() -> None:
    """Route routing decisions to their own file so they are greppable in one place.

    Without this, INFO-level routing logs only reach agent.log (per-session) and
    WARNING reaches errors.log; they never appear in gateway.log, which makes
    "did the router actually switch" hard to answer from the container logs.
    """
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    log_dir = os.path.join(hermes_home, "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "model-picker.log"))
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _rank(name: str) -> int:
    return RANK.get(name, RANK[_MID])


def _higher(name: str) -> str:
    i = _rank(name)
    nxt = NAMES[min(i + 1, len(NAMES) - 1)]
    if _rank(nxt) > _rank(_ESCALATE_MAX):
        return _ESCALATE_MAX
    return nxt


def _escalation_threshold(name: str) -> int:
    return _ESCALATION_ERRORS.get(name, 3)


# A real tool error has a non-empty "error" value or "failed": true.
# Successful results carry "error": null / "" or "failed": false and must not
# count toward escalation, otherwise two clean tool calls false-escalate.
_ERROR_PAT = re.compile(r'"(?:error|failed)"\s*:\s*(?!\s*null\b)(?!\s*false\b)(?!\s*"")')

# Tier tokens a message may name: canonical low/default/high plus the
# deprecated "medium" alias, which as_name() resolves to "default".
_TIER_WORD = "|".join((*NAMES, *LEGACY_NAMES))
# Bare mentions are restricted to the unambiguous tokens. A bare "default" is
# ordinary English ("the default is fine") and must not pin a tier, so the
# canonical name pins only via the /default slash form or an explicit phrase.
_BARE_WORD = "|".join(n for n in (*NAMES, *LEGACY_NAMES) if n != "default")

_NAME_RE = re.compile(
    rf"(?:^|(?<=\s)|(?<=\())/?({_BARE_WORD})(?:\b|(?=\)))",
    re.IGNORECASE,
)
_ACK_RE = re.compile(
    r"^(ok|okay|thanks|thank you|thx|got it|understood|sure|yes|no|yep|nope|"
    r"alright|cool|great|nice|perfect|done|noted|ack|hello|hi|hey)"
    r"[!?.]*$",
    re.IGNORECASE,
)
# WebUI prefixes every user turn; strip before ack/length/sentence heuristics.
_WEBUI_WORKSPACE_RE = re.compile(
    r"^\[Workspace::v1:\s*[^\]]+\]\s*",
    re.IGNORECASE,
)
# Slash pin only at the start of the message (after the WebUI workspace prefix).
# Mid-paragraph "/high" in a bug report must not route to a stronger model.
_SLASH_PIN_RE = re.compile(
    rf"^/({_TIER_WORD})\b",
    re.IGNORECASE,
)
# Phrase pins ("pin high", "please use default") — only honoured on short messages.
_PIN_PHRASE_RE = re.compile(
    r"(?:^|\s)(?:use|pin|switch\s+to|run\s+(?:on|at)|please\s+use)\s+/?"
    rf"({_TIER_WORD})\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+")

_lock = threading.Lock()
_last_user_sid: str = ""  # session of the most recent real user turn (command anchor)
_pinned: dict[str, bool] = {}
_last_tier: dict[str, str] = {}
_last_msg: dict[str, tuple[str, str]] = {}  # (msg, tier) to skip re-classifying a repeat
_tool_errors: dict[str, int] = {}
_checkpoint: dict[str, bool] = {}  # session -> escalation checkpoint nudge pending
_compact: dict[str, bool] = {}  # session -> one-shot forced compaction at next boundary
_handoff: dict[str, dict[str, Any]] = {}  # session -> handoff for the context engine
_refused: dict[str, str] = {}  # session -> provider a tier was refused for (log once)

# Injected into the next tool-continuation turn when an escalation checkpoint
# is pending. The working model (not the classifier) decides whether to call
# escalate_model — this is just a nudge.
_CHECKPOINT_NUDGE = (
    "⚠️ Repeated tool failures on this task. If you are stuck, you may call the "
    "`escalate_model` tool to hand off to a stronger model. Provide a summary of "
    "the conversation, the current task and intent, what you already tried, where "
    "it failed (include the exact error), and your best next hypothesis. Otherwise, "
    "continue working normally."
)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _strip_platform_prefix(msg: str) -> str:
    return _WEBUI_WORKSPACE_RE.sub("", msg or "")


def _sentence_count(msg: str) -> int:
    parts = [p for p in _SENTENCE_SPLIT_RE.split(msg or "") if p.strip()]
    return max(1, len(parts))


def _provider_matches(want: str, got: str) -> bool:
    """True when a tier's provider can ride the session's provider.

    The request's client is built from the session provider, so the two must be
    the same API host. Provider ids carry suffixes and aliases in practice
    ("deepseek" / "deepseek-chat"), so a containment test is the honest check;
    unknown values (empty) pass and the host decides.
    """
    a, b = _norm(want), _norm(got)
    if not a or not b:
        return True
    return a == b or a in b or b in a


def _tier_model(name: str) -> str:
    meta = MODELS.get(name) or {}
    return str(meta.get("model") or "").strip()


def _tier_provider(name: str) -> str:
    meta = MODELS.get(name) or {}
    return str(meta.get("provider") or "").strip()


def _current_tier(session_id: str) -> str | None:
    with _lock:
        return _last_tier.get(session_id or "")


def _token_to_name(*parts: str | None) -> str | None:
    for part in parts:
        if not part:
            continue
        name = as_name(part)
        if name:
            return name
    return None


def _detect_explicit_tier(msg: str) -> str | None:
    """Honor slash-at-start and short pin phrases — not mid-paragraph /high."""
    text = _strip_platform_prefix(msg)
    m = _SLASH_PIN_RE.match(text)
    if m:
        return as_name(m.group(1))

    words = text.split()
    if len(words) <= 8:
        reqs: set[str] = set()
        for hit in _PIN_PHRASE_RE.finditer(text):
            name = as_name(hit.group(1))
            if name:
                reqs.add(name)
        if reqs:
            return max(reqs, key=_rank)

    mentions: set[str] = set()
    for hit in _NAME_RE.finditer(text):
        name = _token_to_name(*hit.groups())
        if name:
            mentions.add(name)
    if len(mentions) != 1:
        return None
    # Short messages like "medium please" / "high" only. A bare "default" is
    # not a mention — see _NAME_RE.
    if len(words) <= 6:
        return next(iter(mentions))
    return None


def _classifier_text(response: Any) -> str:
    """Read the completion text off whatever the host facade returned."""
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _classifier_complete(messages: list[dict[str, Any]]) -> str:
    """One bounded completion through the host-owned plugin LLM facade.

    ``ctx.llm`` overrides are fail-closed: a model/provider the host has not
    allowed for this plugin raises, in which case the same request is retried
    on the active model rather than losing the classification.
    """
    llm = getattr(_ctx, "llm", None) if _ctx is not None else None
    if llm is None:
        raise RuntimeError("no ctx.llm facade")
    call: dict[str, Any] = {
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0.0,
        # Hard cap so this advisory call can never eat the gateway's 30s hook
        # budget (pre_llm_call). On timeout _classify fails open to low — the
        # turn proceeds, escalation corrects a miss later.
        "timeout": _CLASSIFIER_TIMEOUT_S,
    }
    overrides: dict[str, Any] = {}
    if _CLASSIFIER_MODEL:
        overrides["model"] = _CLASSIFIER_MODEL
    if _CLASSIFIER_PROVIDER:
        overrides["provider"] = _CLASSIFIER_PROVIDER
    try:
        return _classifier_text(llm.complete(**call, **overrides))
    except Exception:
        if not overrides:
            raise
        logger.warning(
            "model-picker: classifier override %s/%s refused by the host policy; "
            "using the active model",
            _CLASSIFIER_PROVIDER or "-",
            _CLASSIFIER_MODEL or "-",
        )
        return _classifier_text(llm.complete(**call))


def _classify(user_message: str, history: list, session_id: str = "") -> str:
    """Return low|default|high. Fail-open to low — escalation corrects a miss.

    The classifier runs on the host's plugin-LLM route (the session's active
    model, or the override the operator allowed for this plugin). It is given
    the previous tier as a signal so it can prefer to stay put (low/default
    only; high is never sticky).
    """
    try:
        with _lock:
            prev_tier = _last_tier.get(session_id)
        body = _strip_platform_prefix(user_message)
        system = _CLASSIFIER
        if prev_tier is not None:
            system = f"Previous turn tier: {prev_tier}.\n\n" + _CLASSIFIER
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        ctx_parts: list[str] = []
        budget = _CLASSIFIER_CONTEXT_CHARS
        for msg in reversed(history or []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("assistant", "user"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c))
                    for c in content
                )
            if not isinstance(content, str):
                content = str(content)
            text = content.strip()
            if not text:
                continue
            ctx_parts.insert(0, f"{role}: {text}")
            budget -= len(text)
            if budget <= 0:
                break
        if ctx_parts:
            messages.append(
                {"role": "user", "content": "[Conversation context]\n" + "\n---\n".join(ctx_parts)}
            )
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": body[:800] or user_message[:800]})

        raw = _classifier_complete(messages)
        raw_l = raw.lower()
        allowed = (_MIN, _MID, _TOP)
        found = [n for n in allowed if re.search(rf"\b{n}\b", raw_l)]
        if len(found) == 1:
            return found[0]
        named = as_name(raw)
        if named in allowed:
            return named
        logger.warning("model-picker: classifier non-name %r — default low", raw[:40])
    except Exception as exc:
        logger.warning("model-picker: classifier failed (%s) — default low", exc)
    return _MIN


def _target_tier(session_id: str, msg: str, history: list) -> tuple[str, str]:
    """Compute (tier, reason) for a real user turn.

    Repeats of the last classified message reuse its tier without a second
    triage call. No sentence/word floor — weighting is the classifier prior.
    """
    with _lock:
        cached = _last_msg.get(session_id)
    if cached is not None and cached[0] == msg:
        return cached[1], "cached"

    body = _strip_platform_prefix(msg)
    words = body.split()
    n_sent = _sentence_count(body)
    explicit = _detect_explicit_tier(msg)
    if explicit is not None:
        name, reason = explicit, "explicit"
    elif _ACK_RE.match(body) and len(words) <= 6:
        name, reason = _MIN, "ack"
    else:
        name = _classify(msg, history, session_id)
        reason = "classify"

    logger.info(
        "model-picker: route %s (%s) words=%d sentences=%d preview=%r",
        name,
        reason,
        len(words),
        n_sent,
        body[:120],
    )

    with _lock:
        _last_msg[session_id] = (msg, name)
    return name, reason


def _set_tier(session_id: str, name: str, reason: str) -> None:
    """Record the tier for the session. The request middleware applies it."""
    with _lock:
        prev = _last_tier.get(session_id)
        _last_tier[session_id] = name
    if prev != name:
        logger.info(
            "model-picker: %s (%s) sid=%s model=%s -> %s",
            name,
            reason,
            session_id or "-",
            _tier_model(name) or "-",
            _tier_provider(name) or "-",
        )


def _take_compact(session_id: str) -> bool:
    """Consume a pending forced-compaction request (called by the engine)."""
    with _lock:
        return bool(_compact.pop(session_id or "", False))


def _take_handoff(session_id: str) -> dict[str, Any] | None:
    """Consume a pending escalation handoff (called by the engine)."""
    with _lock:
        return _handoff.pop(session_id or "", None)


def _should_skip(platform: str, kwargs: dict) -> bool:
    plat = (platform or "").strip().lower()
    if plat in _SKIP_PLATFORMS:
        return True
    # subagent = delegate_task children: model is pinned by delegation config.
    if kwargs.get("parent_session_id") or "":
        return True
    return False


def on_llm_request(
    *,
    request: dict | None = None,
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    **kwargs: Any,
) -> dict | None:
    """Rewrite the outgoing request's model to the session's tier.

    Public middleware surface (``llm_request``): the payload handed in is the
    mutable provider kwargs, and returning ``{"request": {...}}`` replaces it.
    Only the ``model`` key is touched; the API host, auth and every other key
    come from the session unchanged.
    """
    try:
        if not _CONFIGURED or not isinstance(request, dict):
            return None
        if _should_skip(platform, kwargs):
            return None
        sid = session_id or ""
        tier = _current_tier(sid)
        if not tier:
            return None
        want = _tier_model(tier)
        if not want:
            return None
        current = str(request.get("model") or "")
        # Only rewrite the model Hermes itself chose for this session. A
        # different value means a fallback/rotation already switched it, and the
        # fallback's choice outranks the tier.
        ctx_model = str(model or "")
        if ctx_model and current and current != ctx_model:
            logger.debug(
                "model-picker: leaving %s in place (session model is %s)", current, ctx_model
            )
            return None
        if current == want:
            return None
        want_provider = _tier_provider(tier)
        if not _provider_matches(want_provider, provider):
            with _lock:
                already = _refused.get(sid)
                _refused[sid] = want_provider
            if already != want_provider:
                logger.warning(
                    "model-picker: tier %s names provider %s but the session runs %s — "
                    "kept the session model %s (a request middleware cannot move the API "
                    "host; configure all three tiers on one provider)",
                    tier,
                    want_provider,
                    provider,
                    current or "-",
                )
            return None
        new_request = dict(request)
        new_request["model"] = want
        with _lock:
            _refused.pop(sid, None)
        return {
            "request": new_request,
            "name": "model-picker",
            "reason": f"tier={tier}",
        }
    except Exception as exc:
        logger.warning("model-picker: on_llm_request error: %s", exc, exc_info=True)
        return None


def on_pre_llm_call(
    *,
    user_message: str = "",
    conversation_history: list | None = None,
    model: str = "",
    session_id: str = "",
    platform: str = "",
    **kwargs: Any,
) -> dict | None:
    """Rate the turn and record its tier. Applied later by ``on_llm_request``."""
    global _last_user_sid
    try:
        if not _CONFIGURED:
            return None
        if _should_skip(platform, kwargs):
            return None
        sid = session_id or ""
        if (user_message or "").strip():
            # A real user turn anchors which session the human is talking in;
            # slash commands resolve against this.
            with _lock:
                _last_user_sid = sid

        with _lock:
            pinned = _pinned.get(sid, False)
            current = _last_tier.get(sid)
            checkpoint = _checkpoint.get(sid, False)

        if pinned:
            if checkpoint:
                with _lock:
                    _checkpoint[sid] = False
            return None

        msg = (user_message or "").strip()
        if not msg:
            # Empty hook payload (tool-call continuation): keep the route
            # coherent, and if an escalation checkpoint is pending, nudge the
            # working model toward escalate_model (one-shot).
            if checkpoint:
                with _lock:
                    _checkpoint[sid] = False
                return {"context": _CHECKPOINT_NUDGE}
            return None

        name, reason = _target_tier(sid, msg, conversation_history or [])
        # A classifier-driven tier change swaps models; compact the transcript
        # at the turn boundary so the incoming model cold-reads a summary, not
        # the full history. Explicit pins and escalation take other paths.
        if reason == "classify" and current is not None and current != name:
            with _lock:
                _compact[sid] = True
        with _lock:
            _tool_errors[sid] = 0
            _checkpoint[sid] = False
            # Slash/phrase pins arriving as a user message (WebUI buttons send
            # "/high" through the composer) must stick, not one-shot.
            if reason == "explicit":
                _pinned[sid] = True
        _set_tier(sid, name, reason)
        return None
    except Exception as exc:
        logger.warning("model-picker: on_pre_llm_call error: %s", exc, exc_info=True)
        return None


def on_post_tool_call(
    *,
    tool_name: str = "",
    result: str | None = None,
    session_id: str = "",
    **kwargs: Any,
) -> None:
    """Count consecutive tool errors; at threshold, stage an escalation checkpoint."""
    try:
        sid = session_id or ""
        if not sid:
            return
        with _lock:
            if _pinned.get(sid, False):
                return

        is_error = False
        if result is not None:
            # Hook may pass dict/list/ToolResult — never slice non-str.
            if isinstance(result, str):
                head_src = result
            else:
                try:
                    import json as _json

                    head_src = _json.dumps(result, default=str)
                except Exception:
                    head_src = str(result)
            head = head_src[:500].lower()
            if (
                _ERROR_PAT.search(head)
                or head_src.startswith("Error")
                or (
                    '"exit_code": ' in head
                    and '"exit_code": 0' not in head
                    and '"exit_code": null' not in head
                )
            ):
                is_error = True

        with _lock:
            if is_error:
                _tool_errors[sid] = _tool_errors.get(sid, 0) + 1
            else:
                _tool_errors[sid] = 0
            count = _tool_errors.get(sid, 0)
            current = _last_tier.get(sid, _MIN)

        threshold = _escalation_threshold(current)
        if is_error and count >= threshold and _rank(current) < _rank(_ESCALATE_MAX):
            # Stage an escalation checkpoint instead of climbing directly: the
            # working model decides whether to escalate via escalate_model.
            with _lock:
                _checkpoint[sid] = True
                _tool_errors[sid] = 0
            logger.info(
                "model-picker: escalation checkpoint staged sid=%s after %d tool errors (tier=%s)",
                sid,
                count,
                current,
            )
    except Exception as exc:
        logger.warning("model-picker: on_post_tool_call error: %s", exc, exc_info=True)


def _resolve_cmd_sid() -> str:
    """Which session a slash command should target.

    Slash commands are dispatched with only raw_args — no session id — so the
    plugin infers the session from the most recent real user turn. A pin typed
    as the very first message of a session falls back to the empty session;
    ``/high`` sent through the composer instead arrives as a user turn and is
    detected by ``pre_llm_call`` with the real session id.
    """
    with _lock:
        return _last_user_sid or ""


def _cmd_pin(raw_args: str, name: str) -> str:
    del raw_args
    sid = _resolve_cmd_sid()
    meta = MODELS[name]
    with _lock:
        _pinned[sid] = True
    logger.info("model-picker: /%s pin sid=%s", name, sid or "-")
    _set_tier(sid, name, "pin")
    scope = "this session" if sid else "the next user turn of this session"
    return (
        f"Pinned to {meta['label']} ({meta['provider']} / {meta['model']}) for {scope}. "
        "Auto-routing paused. /auto to resume."
    )


def _cmd_auto(raw_args: str) -> str:
    del raw_args
    sid = _resolve_cmd_sid()
    with _lock:
        was = _pinned.pop(sid, False)
        # Drop cached tier + message so the next turn is classified fresh.
        _last_msg.pop(sid, None)
        _last_tier.pop(sid, None)
        _tool_errors.pop(sid, None)
        _compact.pop(sid, None)
    logger.info("model-picker: /auto sid=%s was_pinned=%s", sid or "-", was)
    if was:
        return "Auto routing resumed. Next turn is classified automatically."
    return "Auto routing already active."


ESCALATE_SCHEMA: dict[str, Any] = {
    "name": "escalate_model",
    "description": (
        "Escalate to the next-higher model when the current model is stuck. "
        "Provide a structured handoff so the stronger model can continue without "
        "re-reading the full conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise summary of what has been established/decided so far.",
            },
            "task_state": {
                "type": "string",
                "description": "What this turn is trying to accomplish, in detail.",
            },
            "tried_so_far": {
                "type": "string",
                "description": "What approaches were already attempted.",
            },
            "failure_point": {
                "type": "string",
                "description": "Where it is stuck, including the exact error text if any.",
            },
            "next_hypothesis": {
                "type": "string",
                "description": "Best next approach to try on the stronger model (optional).",
            },
        },
        "required": ["summary", "task_state", "failure_point"],
    },
}


def _handle_escalate_model(args: dict | None = None, **kwargs: Any) -> str:
    """Switch to the next tier and stash a structured handoff for the context engine.

    Hermes dispatches tools as ``handler(args, **context)``: the model's
    arguments arrive in ``args``; ``session_id``/``task_id`` arrive as kwargs.
    """
    args = args if isinstance(args, dict) else {}
    try:
        sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "").strip()
        if not sid:
            sid = _resolve_cmd_sid()
        with _lock:
            pinned = _pinned.get(sid, False)
            current = _last_tier.get(sid, _MIN)
        if pinned:
            return "Pinned session — escalation disabled. /auto to resume auto-routing."
        if _rank(current) >= _rank(_ESCALATE_MAX):
            return f"Already at the highest tier ({_ESCALATE_MAX}); nothing to escalate to."

        target = _higher(current)
        dest = MODELS[target]
        handoff = {
            "from_tier": current,
            "to_tier": target,
            "to_model": dest.get("model") or target,
            "summary": str(args.get("summary") or "").strip(),
            "task_state": str(args.get("task_state") or "").strip(),
            "tried_so_far": str(args.get("tried_so_far") or "").strip(),
            "failure_point": str(args.get("failure_point") or "").strip(),
            "next_hypothesis": str(args.get("next_hypothesis") or "").strip(),
        }
        with _lock:
            _handoff[sid] = handoff
        _set_tier(sid, target, "escalate_model")
        with _lock:
            _tool_errors[sid] = 0
            _checkpoint[sid] = False
        meta = MODELS[target]
        note = (
            "The stronger model reads a compact handoff instead of the full "
            "conversation."
            if _engine is not None
            else "The model changed; the transcript is unchanged (the handoff "
            "context engine is not active on this host)."
        )
        return (
            f"Escalated to {meta['label']} ({meta.get('provider')}/{meta.get('model')}). "
            + note
        )
    except Exception as exc:
        logger.warning("model-picker: escalate_model failed: %s", exc, exc_info=True)
        return f"escalate_model failed: {exc}"


def _register_escalate_tool(ctx: Any) -> None:
    ctx.register_tool(
        name="escalate_model",
        toolset="plugin",
        schema=ESCALATE_SCHEMA,
        handler=_handle_escalate_model,
        description=ESCALATE_SCHEMA["description"],
        emoji="🪜",
    )


def _host_engine_name() -> str:
    """``context.engine`` from the host's raw config, or "" when unreadable.

    The host picks a plugin context engine by this name; registering an engine
    the config does not name would occupy the single public engine slot for
    nothing, so the registration is gated on it.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
    except Exception as exc:
        logger.warning("model-picker: cannot read the host config (%s)", exc)
        return ""
    if not isinstance(cfg, dict):
        return ""
    context = cfg.get("context")
    if not isinstance(context, dict):
        return ""
    return str(context.get("engine") or "").strip()


def _load_engine_module() -> Any:
    try:
        from . import engine as engine_mod

        return engine_mod
    except ImportError:
        path = Path(__file__).with_name("engine.py")
        spec = importlib.util.spec_from_file_location("model_picker_engine", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _register_engine(ctx: Any, name: str) -> Any | None:
    """Register the handoff engine — only when the host config names it."""
    if not hasattr(ctx, "register_context_engine"):
        logger.warning("model-picker: ctx has no register_context_engine; handoff disabled")
        return None
    if name != _s.ENGINE_NAME:
        logger.info(
            "model-picker: handoff engine not registered (host context.engine=%r, "
            "not %r)",
            name or "",
            _s.ENGINE_NAME,
        )
        return None
    try:
        engine_mod = _load_engine_module()
    except Exception as exc:
        logger.warning("model-picker: handoff engine import failed: %s", exc)
        return None
    engine_mod.bind_router(take_handoff=_take_handoff, take_compact=_take_compact)
    try:
        inst = engine_mod.ModelPickerContextEngine(model="")
        ctx.register_context_engine(inst)
    except Exception as exc:
        logger.warning("model-picker: handoff engine registration failed: %s", exc)
        return None
    logger.info("model-picker: handoff context engine registered (name=%s)", inst.name)
    return inst


def register(ctx: Any) -> None:
    """Plugin entry point. Public surfaces only — see the module docstring."""
    global _ctx, _engine
    _ctx = ctx
    _attach_file_handler()
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_middleware("llm_request", on_llm_request)
    _register_escalate_tool(ctx)
    _engine = _register_engine(ctx, _host_engine_name())
    for name in NAMES:
        meta = MODELS[name]
        label = meta.get("label") or name.capitalize()
        ctx.register_command(
            name,
            lambda args, n=name: _cmd_pin(args, n),
            f"Pin session to {label} ({meta.get('provider')}/{meta.get('model')})",
        )
    ctx.register_command("auto", _cmd_auto, "Resume model-picker auto routing")
    if not _CONFIGURED:
        logger.warning(
            "model-picker: no model id for %s, and the Hermes config names no "
            "primary model either; plugin is registered but inert. Set "
            "MODEL_PICKER_*_MODEL, config.json, or hermesPnP.models.",
            ", ".join(_UNCONFIGURED_SLOTS) or "any tier",
        )
        return
    labels = " / ".join(f"{n} {MODELS[n].get('label')}" for n in NAMES)
    logger.info(
        "model-picker: %s | escalate≤%s | /low /default /high /auto | llm_request middleware",
        labels,
        _ESCALATE_MAX,
    )
