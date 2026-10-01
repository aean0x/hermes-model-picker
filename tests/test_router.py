"""Router pin/name helpers, the llm_request middleware, and the registered surface.

No Hermes install is needed for these: the middleware and the registration shape
are pure plugin code. The engine-registration cases skip when Hermes is absent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:  # loaded as ``tests.test_router`` under discovery
    from .support import install_engine_stub
except ImportError:  # loaded as a top-level module
    from support import install_engine_stub  # type: ignore

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

try:
    import agent.context_compressor  # noqa: F401

    _HERMES_IMPORTABLE = True
except Exception:
    _HERMES_IMPORTABLE = False


def _host_engine(name: str):
    """Stub the host config's context.engine value."""
    pkg = types.ModuleType("hermes_cli")
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.read_raw_config = lambda: {"context": {"engine": name}}
    pkg.config = cfg_mod  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"hermes_cli": pkg, "hermes_cli.config": cfg_mod})


def _load():
    os.environ["MODEL_PICKER_CONFIG"] = str(ROOT / "tests" / "oobe-ids.json")
    spec = importlib.util.spec_from_file_location("model_picker_mod", ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCtx:
    """Minimal public PluginContext surface (mirrors the shipped capability probe)."""

    def __init__(self) -> None:
        self.hooks: list[str] = []
        self.middleware: list[str] = []
        self.tools: list[str] = []
        self.commands: list[str] = []
        self.engines: list[object] = []

    def register_hook(self, name, callback):
        self.hooks.append(str(name))

    def register_middleware(self, kind, callback):
        self.middleware.append(str(kind))

    def register_tool(self, name, *args, **kwargs):
        self.tools.append(str(name))

    def register_command(self, name, *args, **kwargs):
        self.commands.append(str(name))

    def register_context_engine(self, engine):
        self.engines.append(engine)


class Pins(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_named_slash_pins(self) -> None:
        self.assertEqual(self.mod._detect_explicit_tier("/low"), "low")
        self.assertEqual(self.mod._detect_explicit_tier("/default"), "default")
        self.assertEqual(self.mod._detect_explicit_tier("/high please"), "high")

    def test_deprecated_medium_slash_pins_default(self) -> None:
        self.assertEqual(self.mod._detect_explicit_tier("/medium"), "default")

    def test_use_phrase(self) -> None:
        self.assertEqual(self.mod._detect_explicit_tier("please use default"), "default")
        self.assertEqual(self.mod._detect_explicit_tier("please use medium"), "default")
        self.assertEqual(self.mod._detect_explicit_tier("pin high"), "high")

    def test_bare_name_in_long_critique_is_not_a_pin(self) -> None:
        msg = "The low estimate in the budget is wrong because the medium path is already over cost."
        self.assertIsNone(self.mod._detect_explicit_tier(msg))

    def test_short_bare_name_is_a_pin(self) -> None:
        self.assertEqual(self.mod._detect_explicit_tier("medium"), "default")
        self.assertEqual(self.mod._detect_explicit_tier("low please"), "low")

    def test_bare_default_is_not_a_pin(self) -> None:
        # "default" is ordinary English; only /default or "pin default" pins it.
        self.assertIsNone(self.mod._detect_explicit_tier("is that the default?"))
        self.assertIsNone(self.mod._detect_explicit_tier("default"))
        self.assertEqual(self.mod._detect_explicit_tier("pin default"), "default")

    def test_mid_paragraph_slash_high_is_not_a_pin(self) -> None:
        msg = (
            "Agree with your assessment. Execute in a PR please. "
            "Double check the session where I did an explicit /high "
            "and it finished as deepseek pro."
        )
        self.assertIsNone(self.mod._detect_explicit_tier(msg))


class Names(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_as_name_named_only(self) -> None:
        self.assertEqual(self.mod.as_name("low"), "low")
        self.assertEqual(self.mod.as_name("HIGH"), "high")
        self.assertEqual(self.mod.as_name("default"), "default")
        self.assertEqual(self.mod.as_name("medium"), "default")
        self.assertIsNone(self.mod.as_name("ultra"))

    def test_higher_climbs_to_high(self) -> None:
        self.assertEqual(self.mod._higher("low"), "default")
        self.assertEqual(self.mod._higher("default"), "high")
        self.assertEqual(self.mod._higher("high"), "high")

    def test_router_does_not_touch_reasoning_config(self) -> None:
        src = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("reasoning_config", src)
        self.assertNotIn(".reasoning_effort", src)


class Middleware(unittest.TestCase):
    """llm_request is the only place the model changes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def setUp(self) -> None:
        with self.mod._lock:
            self.mod._last_tier.clear()
            self.mod._refused.clear()
            self.mod._pinned.clear()

    def _tier_provider(self, tier: str) -> str:
        return str(self.mod.MODELS[tier]["provider"])

    def test_rewrites_only_the_model(self) -> None:
        tier = "high"
        with self.mod._lock:
            self.mod._last_tier["s1"] = tier
        request = {
            "model": "some-session-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        }
        out = self.mod.on_llm_request(
            request=request,
            session_id="s1",
            platform="cli",
            model="some-session-model",
            provider=self._tier_provider(tier),
        )
        self.assertIsNotNone(out)
        assert out is not None
        new_request = out["request"]
        self.assertEqual(new_request["model"], self.mod.MODELS[tier]["model"])
        self.assertEqual(new_request["temperature"], 0.2)
        self.assertEqual(new_request["messages"], request["messages"])
        # The payload handed in is never mutated in place.
        self.assertEqual(request["model"], "some-session-model")

    def test_no_tier_no_rewrite(self) -> None:
        self.assertIsNone(
            self.mod.on_llm_request(
                request={"model": "m"},
                session_id="untouched",
                platform="cli",
                model="m",
                provider=self._tier_provider("low"),
            )
        )

    def test_matching_model_is_not_replaced(self) -> None:
        tier = "low"
        model = self.mod.MODELS[tier]["model"]
        with self.mod._lock:
            self.mod._last_tier["s2"] = tier
        self.assertIsNone(
            self.mod.on_llm_request(
                request={"model": model},
                session_id="s2",
                platform="cli",
                model=model,
                provider=self._tier_provider(tier),
            )
        )

    def test_foreign_provider_is_refused(self) -> None:
        tier = "high"
        with self.mod._lock:
            self.mod._last_tier["s3"] = tier
        out = self.mod.on_llm_request(
            request={"model": "session-model"},
            session_id="s3",
            platform="cli",
            model="session-model",
            provider="an-unrelated-provider",
            base_url="https://example.invalid/v1",
        )
        self.assertIsNone(out)
        with self.mod._lock:
            self.assertEqual(self.mod._refused.get("s3"), self._tier_provider(tier))

    def test_fallback_model_outranks_the_tier(self) -> None:
        # A fallback/rotation already changed the payload's model; the tier must
        # not fight it back.
        with self.mod._lock:
            self.mod._last_tier["s4"] = "high"
        self.assertIsNone(
            self.mod.on_llm_request(
                request={"model": "fallback-model"},
                session_id="s4",
                platform="cli",
                model="session-model",
                provider=self._tier_provider("high"),
            )
        )

    def test_cron_and_subagent_platforms_are_skipped(self) -> None:
        with self.mod._lock:
            self.mod._last_tier["s5"] = "high"
        for platform, kwargs in (("cron", {}), ("cli", {"parent_session_id": "parent"})):
            self.assertIsNone(
                self.mod.on_llm_request(
                    request={"model": "session-model"},
                    session_id="s5",
                    platform=platform,
                    model="session-model",
                    provider=self._tier_provider("high"),
                    **kwargs,
                )
            )


class RegisteredSurface(unittest.TestCase):
    """The plugin extends Hermes through public surfaces only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def _register(self, host_engine: str = "compressor"):
        ctx = FakeCtx()
        with _host_engine(host_engine):
            self.mod.register(ctx)
        return ctx

    def test_declared_surface_matches_plugin_yaml(self) -> None:
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        for hook in ("pre_llm_call", "post_tool_call"):
            self.assertIn(hook, manifest)
        self.assertIn("llm_request", manifest)
        self.assertNotIn("pre_api_request", manifest)

    def test_registers_hooks_middleware_tool_and_commands(self) -> None:
        ctx = self._register()
        self.assertEqual(sorted(ctx.hooks), ["post_tool_call", "pre_llm_call"])
        self.assertEqual(ctx.middleware, ["llm_request"])
        self.assertEqual(ctx.tools, ["escalate_model"])
        self.assertEqual(sorted(ctx.commands), ["auto", "default", "high", "low"])

    def test_no_internals_are_touched(self) -> None:
        src = (ROOT / "__init__.py").read_text(encoding="utf-8")
        for banned in (
            "_replace_primary_openai_client",
            "_client_kwargs",
            "setattr(",
            "MutationObserver",
            "_install_agent_capture",
            "_wrap_cls_method",
            "switch_model(",
            "_cli_ref",
            "auxiliary_client",
        ):
            self.assertNotIn(banned, src)

    @unittest.skipUnless(HERMES_SRC.is_dir(), "needs a Hermes checkout")
    def test_engine_registers_only_when_the_host_names_it(self) -> None:
        if str(HERMES_SRC) not in sys.path:
            sys.path.insert(0, str(HERMES_SRC))
        install_engine_stub()
        self.assertEqual(len(self._register("model-picker").engines), 1)
        self.assertEqual(self._register("compressor").engines, [])


if __name__ == "__main__":
    unittest.main()
