"""Turn-start routing, escalation, the classifier call and the handoff engine."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:  # loaded as ``tests.test_routing`` under discovery
    from .support import engine_base_is_stubbed, install_engine_stub
except ImportError:  # loaded as a top-level module
    from support import engine_base_is_stubbed, install_engine_stub  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def _hermes_src() -> Path:
    """A Hermes source checkout, when one exists (set HERMES_SRC to point at it)."""
    env = os.environ.get("HERMES_SRC", "").strip()
    if env:
        return Path(env)
    for rel in ("src/hermes-agent", "hermes-agent"):
        cand = Path.home() / rel
        if cand.is_dir():
            return cand
    return Path.home() / "src" / "hermes-agent"


HERMES_SRC = _hermes_src()

# engine.py imports agent.context_compressor, which exists only inside a Hermes
# install. Tests that drive the engine skip when Hermes is not importable.
try:
    import agent.context_compressor  # noqa: F401

    _HERMES_IMPORTABLE = True
except Exception:
    _HERMES_IMPORTABLE = False


def _load():
    os.environ["MODEL_PICKER_CONFIG"] = str(ROOT / "tests" / "oobe-ids.json")
    spec = importlib.util.spec_from_file_location("model_picker_routing", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_ctx(complete):
    """A PluginContext stub whose only used surface is the llm facade."""

    class FakeLlm:
        def complete(self, messages=None, **kwargs):
            return complete(messages=messages, **kwargs)

    return SimpleNamespace(llm=FakeLlm())


class TargetTier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def setUp(self) -> None:
        with self.mod._lock:
            self.mod._last_msg.clear()
            self.mod._last_tier.clear()
            self.mod._pinned.clear()
            self.mod._checkpoint.clear()
            self.mod._tool_errors.clear()

    def test_explicit_high_is_not_clamped(self) -> None:
        name, reason = self.mod._target_tier("s1", "/high please", [])
        self.assertEqual(name, "high")
        self.assertEqual(reason, "explicit")

    def test_classify_high_is_accepted(self) -> None:
        with patch.object(self.mod, "_classify", return_value="high"):
            name, reason = self.mod._target_tier(
                "s2", "please execute this monetary transfer now", []
            )
        self.assertEqual(name, "high")
        self.assertEqual(reason, "classify")

    def test_no_floor_on_multi_sentence_low(self) -> None:
        with patch.object(self.mod, "_classify", return_value="low"):
            long = (
                "First sentence is long enough to count. "
                "Second sentence makes this multi-sentence work."
            )
            name, reason = self.mod._target_tier("s3", long, [])
        self.assertEqual(name, "low")
        self.assertEqual(reason, "classify")

    def test_ack_stays_low(self) -> None:
        name, reason = self.mod._target_tier("s4", "ok", [])
        self.assertEqual(name, "low")
        self.assertEqual(reason, "ack")

    def test_cached_message_reuses_tier(self) -> None:
        with self.mod._lock:
            self.mod._last_msg["s5"] = ("same", "default")
        name, reason = self.mod._target_tier("s5", "same", [])
        self.assertEqual(name, "default")
        self.assertEqual(reason, "cached")

    def test_explicit_slash_pins_session(self) -> None:
        self.mod.on_pre_llm_call(
            user_message="/high",
            conversation_history=[],
            session_id="pin1",
        )
        with self.mod._lock:
            self.assertTrue(self.mod._pinned.get("pin1"))
            self.assertEqual(self.mod._last_tier.get("pin1"), "high")

    def test_mid_paragraph_high_does_not_pin(self) -> None:
        msg = (
            "Agree with your assessment. Execute in a PR please.\n\n"
            "check the session where I did an explicit /high and it finished as pro."
        )
        with patch.object(self.mod, "_classify", return_value="default"):
            self.mod.on_pre_llm_call(
                user_message=msg,
                conversation_history=[],
                session_id="pin2",
            )
        with self.mod._lock:
            self.assertFalse(self.mod._pinned.get("pin2", False))
            self.assertEqual(self.mod._last_tier.get("pin2"), "default")


class Escalate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def setUp(self) -> None:
        with self.mod._lock:
            self.mod._last_msg.clear()
            self.mod._last_tier.clear()
            self.mod._pinned.clear()
            self.mod._checkpoint.clear()
            self.mod._tool_errors.clear()
            self.mod._handoff.clear()
            self.mod._compact.clear()
            self.mod._last_user_sid = ""

    def test_higher_ladder(self) -> None:
        self.assertEqual(self.mod._higher("low"), "default")
        self.assertEqual(self.mod._higher("default"), "high")
        self.assertEqual(self.mod._higher("high"), "high")

    def test_pinned_refuses_escalation(self) -> None:
        with self.mod._lock:
            self.mod._pinned["sid"] = True
            self.mod._last_tier["sid"] = "default"
            self.mod._last_user_sid = "sid"
        out = self.mod._handle_escalate_model(
            {"summary": "s", "task_state": "t", "failure_point": "f"},
            session_id="sid",
        )
        self.assertIn("Pinned", out)
        with self.mod._lock:
            self.assertEqual(self.mod._last_tier["sid"], "default")

    def test_already_high_refuses(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["sid"] = "high"
        out = self.mod._handle_escalate_model(
            {"summary": "s", "task_state": "t", "failure_point": "f"},
            session_id="sid",
        )
        self.assertIn("highest tier", out)

    def test_escalate_default_to_high_stashes_handoff(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["sid"] = "default"
        with patch.object(self.mod, "_set_tier") as set_tier:
            # Hermes dispatch shape: handler(args, **context) (tools/registry.py).
            out = self.mod._handle_escalate_model(
                {
                    "summary": "we decided X",
                    "task_state": "trying Y",
                    "tried_so_far": "Z failed",
                    "failure_point": "exact error: boom",
                    "next_hypothesis": "try W",
                },
                session_id="sid",
            )
        self.assertIn("Escalated", out)
        set_tier.assert_called_once_with("sid", "high", "escalate_model")
        handoff = self.mod._take_handoff("sid")
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual(handoff["failure_point"], "exact error: boom")
        self.assertEqual(handoff["summary"], "we decided X")
        self.assertEqual(handoff["from_tier"], "default")
        self.assertEqual(handoff["to_tier"], "high")
        self.assertEqual(handoff["to_model"], self.mod.MODELS["high"]["model"])
        # One-shot: the engine consumes it, and a second read sees nothing.
        self.assertIsNone(self.mod._take_handoff("sid"))

    def test_escalate_schema_is_hermes_tool_shape(self) -> None:
        # Hermes emits {"type": "function", "function": {**schema, "name": ...}},
        # so the argument JSON Schema must live under "parameters".
        schema = self.mod.ESCALATE_SCHEMA
        self.assertEqual(schema["name"], "escalate_model")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertIn("failure_point", schema["parameters"]["properties"])
        self.assertNotIn("properties", schema)

    def test_user_turn_anchors_command_session(self) -> None:
        # /low, /high, /auto resolve via _resolve_cmd_sid; the anchor must be the
        # session that sent the last real user turn.
        with (
            patch.object(self.mod, "_target_tier", return_value=("low", "classify")),
            patch.object(self.mod, "_set_tier"),
        ):
            self.mod.on_pre_llm_call(user_message="hello", session_id="alice")
            self.assertEqual(self.mod._last_user_sid, "alice")
            self.assertEqual(self.mod._resolve_cmd_sid(), "alice")

    def test_auto_after_high_pin_bumps_down(self) -> None:
        with self.mod._lock:
            self.mod._pinned["s"] = True
            self.mod._last_tier["s"] = "high"
            self.mod._last_msg["s"] = ("previous turn", "high")
            self.mod._tool_errors["s"] = 4
            self.mod._compact["s"] = True
            self.mod._last_user_sid = "s"
        out = self.mod._cmd_auto("")
        self.assertIn("resumed", out)
        with self.mod._lock:
            self.assertFalse("s" in self.mod._pinned)
            self.assertFalse("s" in self.mod._last_tier)
            self.assertFalse("s" in self.mod._last_msg)
            self.assertFalse("s" in self.mod._tool_errors)
            self.assertFalse(self.mod._compact.get("s", False))
        with patch.object(self.mod, "_classify", return_value="low"):
            name, reason = self.mod._target_tier(
                "s", "please review this architecture decision carefully", []
            )
        self.assertEqual(name, "low")
        self.assertEqual(reason, "classify")


class ClassifierCall(unittest.TestCase):
    """The classifier runs through the host's plugin-LLM facade."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def setUp(self) -> None:
        self._prev_ctx = self.mod._ctx
        with self.mod._lock:
            self.mod._last_msg.clear()
            self.mod._last_tier.clear()

    def tearDown(self) -> None:
        self.mod._ctx = self._prev_ctx

    def _capture(self, sid: str, user_message: str = "hello there") -> dict:
        captured: dict = {}

        def complete(messages=None, **kwargs):
            captured["messages"] = messages
            captured["timeout"] = kwargs.get("timeout")
            captured["overrides"] = {
                k: v for k, v in kwargs.items() if k in ("model", "provider")
            }
            return SimpleNamespace(text="low")

        self.mod._ctx = _fake_ctx(complete)
        self.mod._classify(user_message, [], sid)
        return captured

    def test_classifier_call_is_bounded(self) -> None:
        # The classifier runs synchronously inside pre_llm_call (30s gateway
        # budget); an unbounded round-trip there timed the hook out.
        captured = self._capture("s-timeout")
        self.assertIsInstance(captured.get("timeout"), float)
        self.assertGreaterEqual(captured["timeout"], 1.0)
        self.assertLessEqual(captured["timeout"], 15.0)

    def test_no_prev_tier_omits_signal(self) -> None:
        captured = self._capture("s1")
        system = captured["messages"][0]["content"]
        self.assertNotIn("Previous turn tier", system)

    def test_prev_tier_is_signalled(self) -> None:
        for sid, tier in (("s2", "default"), ("s3", "high")):
            with self.mod._lock:
                self.mod._last_tier[sid] = tier
            captured = self._capture(sid)
            self.assertIn(f"Previous turn tier: {tier}", captured["messages"][0]["content"])

    def test_no_override_by_default(self) -> None:
        # Without configuration or an operator-allowed override the classifier
        # runs on the session's active model.
        self.assertEqual(self._capture("s4").get("overrides"), {})

    def test_refused_override_retries_without_it(self) -> None:
        captured: dict = {}

        def complete(messages=None, **kwargs):
            captured["overrides"] = {
                k: v for k, v in kwargs.items() if k in ("model", "provider")
            }
            if captured["overrides"]:
                raise RuntimeError("override refused by policy")
            return SimpleNamespace(text="default")

        self.mod._ctx = _fake_ctx(complete)
        with (
            patch.object(self.mod, "_CLASSIFIER_MODEL", "cheap-model"),
            patch.object(self.mod, "_CLASSIFIER_PROVIDER", "somewhere"),
        ):
            tier = self.mod._classify("do a big refactor", [], "s5")
        self.assertEqual(tier, "default")

    def test_missing_facade_fails_open_to_low(self) -> None:
        self.mod._ctx = None
        self.assertEqual(self.mod._classify("anything", [], "s6"), "low")


class ForceCompaction(unittest.TestCase):
    """A classifier-driven tier change requests compaction; other paths don't."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def setUp(self) -> None:
        with self.mod._lock:
            self.mod._last_msg.clear()
            self.mod._last_tier.clear()
            self.mod._pinned.clear()
            self.mod._checkpoint.clear()
            self.mod._tool_errors.clear()
            self.mod._compact.clear()
            self.mod._last_user_sid = ""

    def test_classify_change_requests_compaction(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["s1"] = "low"
        with (
            patch.object(self.mod, "_classify", return_value="default"),
            patch.object(self.mod, "_set_tier"),
        ):
            self.mod.on_pre_llm_call(
                user_message="do a big multi-file refactor",
                conversation_history=[],
                session_id="s1",
            )
        self.assertTrue(self.mod._take_compact("s1"))

    def test_no_change_does_not_request_compaction(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["s2"] = "default"
        with (
            patch.object(self.mod, "_classify", return_value="default"),
            patch.object(self.mod, "_set_tier"),
        ):
            self.mod.on_pre_llm_call(
                user_message="continue where we left off",
                conversation_history=[],
                session_id="s2",
            )
        self.assertFalse(self.mod._take_compact("s2"))

    def test_explicit_pin_does_not_request_compaction(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["s3"] = "low"
        with patch.object(self.mod, "_set_tier"):
            self.mod.on_pre_llm_call(
                user_message="/high please",
                conversation_history=[],
                session_id="s3",
            )
        self.assertFalse(self.mod._take_compact("s3"))

    def test_first_turn_does_not_request_compaction(self) -> None:
        with (
            patch.object(self.mod, "_classify", return_value="default"),
            patch.object(self.mod, "_set_tier"),
        ):
            self.mod.on_pre_llm_call(
                user_message="first message ever",
                conversation_history=[],
                session_id="s4",
            )
        self.assertFalse(self.mod._take_compact("s4"))


@unittest.skipUnless(HERMES_SRC.is_dir(), "needs a Hermes checkout")
class HandoffEngine(unittest.TestCase):
    """The engine reads the router's session state through bind_router."""

    @classmethod
    def setUpClass(cls) -> None:
        if str(HERMES_SRC) not in sys.path:
            sys.path.insert(0, str(HERMES_SRC))
        cls.stubbed_base = not install_engine_stub()
        spec = importlib.util.spec_from_file_location("mr_engine", ROOT / "engine.py")
        assert spec is not None and spec.loader is not None
        cls.eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.eng)

    def _engine(self, handoff=None, compact=False, sid="sid"):
        pending = {"handoff": handoff, "compact": compact}

        def take_handoff(session_id):
            value, pending["handoff"] = pending["handoff"], None
            return value

        def take_compact(session_id):
            value, pending["compact"] = pending["compact"], False
            return value

        self.eng.bind_router(take_handoff=take_handoff, take_compact=take_compact)
        engine = self.eng.ModelPickerContextEngine(model="grok-4.6")
        engine.on_session_start(sid)
        return engine, pending

    def _handoff(self) -> dict:
        return {
            "from_tier": "default",
            "to_tier": "high",
            "to_model": "grok-4.6",
            "summary": "established A",
            "task_state": "doing B",
            "tried_so_far": "C",
            "failure_point": "Error: boom",
            "next_hypothesis": "try D",
        }

    def test_base_class_comes_from_hermes_when_it_imports(self) -> None:
        # The stub only ever stands in where the Hermes runtime is absent; it
        # must never shadow a working import.
        if _HERMES_IMPORTABLE:
            self.assertFalse(engine_base_is_stubbed())
        else:
            self.assertTrue(engine_base_is_stubbed())

    def test_no_handoff_is_noop(self) -> None:
        engine, _ = self._engine()
        req = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        self.assertIsNone(engine.select_context(req))

    def test_handoff_replaces_request_and_keeps_system(self) -> None:
        engine, _ = self._engine(handoff=self._handoff())
        req = [
            {"role": "system", "content": "stable prefix"},
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old asst"},
            {"role": "tool", "content": "Error: boom"},
        ]
        out = engine.select_context(req)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "stable prefix")
        self.assertEqual(out[1]["role"], "user")
        self.assertIn("Error: boom", out[1]["content"])
        self.assertIn("established A", out[1]["content"])
        self.assertIn("default → high (grok-4.6)", out[1]["content"])
        self.assertTrue(any(m.get("content") == "Error: boom" for m in out[2:]))

    def test_handoff_keeps_tool_messages_intact(self) -> None:
        # A rebuilt {"role", "content"} pair dropped tool_calls/tool_call_id and
        # orphaned every tool group in the tail.
        engine, _ = self._engine(handoff=self._handoff())
        calls = [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        req = [
            {"role": "system", "content": "stable prefix"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "Error: boom", "tool_call_id": "call_1", "name": "t"},
        ]
        out = engine.select_context(req)
        assert out is not None
        assistant = [m for m in out if m.get("tool_calls")]
        tool = [m for m in out if m.get("tool_call_id")]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["tool_calls"], calls)
        self.assertEqual(len(tool), 1)
        self.assertEqual(tool[0]["tool_call_id"], "call_1")
        self.assertEqual(tool[0]["name"], "t")

    def test_tail_opens_on_a_turn_boundary(self) -> None:
        engine, _ = self._engine(handoff=self._handoff())
        engine._handoff_tail_chars = 10  # force the budget to cut inside the tail
        req = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "orphan", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "content": "x" * 40, "tool_call_id": "c"},
            {"role": "user", "content": "newest"},
        ]
        out = engine.select_context(req)
        assert out is not None
        tail = out[2:]
        self.assertTrue(tail, "expected a verbatim tail")
        self.assertEqual(tail[0]["role"], "user")

    def test_handoff_is_one_shot(self) -> None:
        engine, _ = self._engine(handoff=self._handoff())
        req = [{"role": "system", "content": "sys"}]
        self.assertIsNotNone(engine.select_context(req))
        self.assertIsNone(engine.select_context(req))

    def test_other_session_gets_no_handoff(self) -> None:
        self.eng.bind_router(
            take_handoff=lambda sid: {} if sid == "someone-else" else None,
            take_compact=lambda sid: False,
        )
        engine = self.eng.ModelPickerContextEngine(model="grok-4.6")
        engine.on_session_start("mine")
        self.assertIsNone(engine.select_context([{"role": "system", "content": "s"}]))

    def test_forced_compaction_is_one_shot_from_the_router(self) -> None:
        engine, pending = self._engine(compact=True)
        self.assertEqual(engine.should_compress_info(0), (True, None))
        self.assertFalse(pending["compact"])
        # Second call falls through to the inherited threshold logic (0 tokens
        # is always under threshold).
        self.assertEqual(engine.should_compress_info(0), (False, None))

    def test_no_router_binding_is_inert(self) -> None:
        self.eng.bind_router(
            take_handoff=lambda sid: None,
            take_compact=lambda sid: False,
        )
        engine = self.eng.ModelPickerContextEngine(model="grok-4.6")
        self.assertEqual(engine.should_compress_info(0), (False, None))
        self.assertIsNone(engine.select_context([{"role": "system", "content": "s"}]))


if __name__ == "__main__":
    unittest.main()
