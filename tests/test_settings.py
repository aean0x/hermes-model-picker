"""Unit tests for model-picker settings (no Hermes)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_OOBE_IDS = ROOT / "tests" / "oobe-ids.json"

_SUFFIXES = (
    "CONFIG",
    "LOW_MODEL",
    "LOW_PROVIDER",
    "LOW_LABEL",
    "LOW_BEST_FOR",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "MEDIUM_MODEL",
    "MEDIUM_PROVIDER",
    "HIGH_MODEL",
    "HIGH_PROVIDER",
    "CLASSIFIER_TIMEOUT_S",
)
# Every generation a test may set, so `_clean_env` never leaks between tests.
_ENV_KEYS = tuple(
    f"{prefix}{suffix}"
    for prefix in ("MODEL_PICKER_",)
    for suffix in _SUFFIXES
)


@contextmanager
def _host_primary(model: str, provider: str):
    """Stub the host's Hermes config so the primary-model fallback is testable."""
    pkg = types.ModuleType("hermes_cli")
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.read_raw_config = lambda: {"model": {"default": model, "provider": provider}}
    pkg.config = cfg_mod  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"hermes_cli": pkg, "hermes_cli.config": cfg_mod}):
        yield


@contextmanager
def _clean_env(**overlay: str):
    keys = set(_ENV_KEYS) | set(overlay)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(overlay)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load(name: str = "mr_settings", *, ids: bool = True):
    if ids and not os.environ.get("MODEL_PICKER_CONFIG"):
        os.environ["MODEL_PICKER_CONFIG"] = str(_OOBE_IDS)
    spec = importlib.util.spec_from_file_location(name, ROOT / "settings.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Defaults(unittest.TestCase):
    def test_three_named_models(self) -> None:
        with _clean_env():
            mod = _load("mr_defaults")
        self.assertEqual(mod.NAMES, ("low", "default", "high"))
        self.assertIn("low", mod.MODELS)
        self.assertIn("default", mod.MODELS)
        self.assertIn("high", mod.MODELS)
        self.assertEqual(mod.ESCALATE_MAX, "high")
        self.assertEqual(mod.ESCALATION_ERRORS["low"], 4)
        self.assertEqual(mod.ESCALATION_ERRORS["default"], 3)
        self.assertNotIn("rocknas", mod.CLASSIFIER.lower())
        self.assertIn("low or default or high", mod.CLASSIFIER)
        self.assertNotIn("ONLY a digit", mod.CLASSIFIER)
        self.assertNotIn("T1", mod.CLASSIFIER)
        self.assertFalse(hasattr(mod, "CLASSIFY_HIGH"))
        self.assertIn("prefer low", mod.CLASSIFIER)
        self.assertIn("Short single-file edits", mod.CLASSIFIER)
        self.assertIn("Small to medium-scoped research", mod.CLASSIFIER)
        self.assertIn("Broad-subject conceptual or deep research", mod.CLASSIFIER)
        self.assertIn("Published outbound voice", mod.CLASSIFIER)
        self.assertIn("in excess of $20", mod.CLASSIFIER)
        self.assertIn("Monetary transactions", mod.CLASSIFIER)
        self.assertNotIn("Architecture", mod.CLASSIFIER)
        self.assertNotIn("Trivial Q&A", mod.CLASSIFIER)
        self.assertNotIn("Rules:", mod.CLASSIFIER)
        cmds = [row["cmd"] for row in mod.webui_models()]
        self.assertEqual(cmds, ["/low", "/default", "/high", "/auto"])
        labels = [row["label"] for row in mod.webui_models()[:3]]
        self.assertEqual(labels, ["Quick", "Standard", "Expert"])

    def test_defaults_match_config_json(self) -> None:
        with _clean_env():
            mod = _load("mr_defaults_json")
        cfg = json.loads((ROOT / "config.default.json").read_text(encoding="utf-8"))
        # The shipped catalog keys ARE the canonical slot names. A catalog that
        # drifts ahead of settings.py ("medium" vs "default") makes
        # _coerce_models_map raise at import and takes the whole plugin down.
        self.assertEqual(list(cfg["models"]), list(mod.NAMES))
        for name in ("low", "default", "high"):
            for key in ("label", "short", "best_for"):
                self.assertEqual(mod.MODELS[name][key], cfg["models"][name][key])
            self.assertNotIn("model", cfg["models"][name])
            self.assertNotIn("provider", cfg["models"][name])
        self.assertNotIn("classify_high", cfg)

    def test_settings_source_has_no_model_ids(self) -> None:
        src = (ROOT / "settings.py").read_text(encoding="utf-8")
        self.assertNotIn("deepseek-v4", src)
        self.assertNotIn("grok-4", src)
        self.assertNotIn("classify_high", src)
        self.assertNotIn("CLASSIFY_HIGH", src)

    def _blank_cfg(self, tmp: str) -> str:
        cfg = Path(tmp) / "cfg.json"
        cfg.write_text(
            json.dumps(
                {
                    "models": {
                        "low": {"model": ""},
                        "default": {"model": ""},
                        "high": {"model": ""},
                    }
                }
            ),
            encoding="utf-8",
        )
        return str(cfg)

    def test_blank_ids_fall_back_to_the_host_primary_model(self) -> None:
        """No plugin config and no env: every tier uses the host's own model."""
        with tempfile.TemporaryDirectory() as tmp:
            with _clean_env(MODEL_PICKER_CONFIG=self._blank_cfg(tmp)), _host_primary(
                "primary-model", "primary-provider"
            ):
                mod = _load("mr_primary", ids=False)
                self.assertTrue(mod.CONFIGURED)
                for name in mod.NAMES:
                    self.assertEqual(mod.MODELS[name]["model"], "primary-model")
                    self.assertEqual(mod.MODELS[name]["provider"], "primary-provider")

    def test_blank_ids_with_no_host_model_are_inert_not_fatal(self) -> None:
        """Refusing to import would kill the plugin on a stock install."""
        with tempfile.TemporaryDirectory() as tmp:
            with _clean_env(MODEL_PICKER_CONFIG=self._blank_cfg(tmp)), _host_primary(
                "", ""
            ):
                mod = _load("mr_inert", ids=False)
        self.assertFalse(mod.CONFIGURED)
        self.assertEqual(set(mod.UNCONFIGURED_SLOTS), set(mod.NAMES))

    def test_declared_models_replace_catalog_ids(self) -> None:
        catalog = json.loads((ROOT / "config.default.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "low": {"model": "cheap", "provider": "p-low"},
                            "default": {"model": "work", "provider": "p-work"},
                            "high": {"model": "voice", "provider": "p-high"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with _clean_env(MODEL_PICKER_CONFIG=str(cfg)):
                mod = _load("mr_declared_overlay")
        self.assertEqual(mod.MODELS["low"]["model"], "cheap")
        self.assertEqual(mod.MODELS["default"]["model"], "work")
        self.assertEqual(mod.MODELS["default"]["provider"], "p-work")
        self.assertEqual(mod.MODELS["high"]["model"], "voice")
        self.assertEqual(mod.MODELS["high"]["provider"], "p-high")
        self.assertEqual(
            mod.MODELS["low"]["best_for"], catalog["models"]["low"]["best_for"]
        )


class LegacyAlias(unittest.TestCase):
    """A v0.8 config / env that says "medium" lands on the "default" slot."""

    def test_medium_model_key_fills_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "low": {"model": "test-low", "provider": "test"},
                            "medium": {"model": "legacy-work", "provider": "test"},
                            "high": {"model": "test-high", "provider": "test"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with _clean_env(MODEL_PICKER_CONFIG=str(cfg)):
                mod = _load("mr_legacy_models_key")
        self.assertEqual(list(mod.MODELS), ["low", "default", "high"])
        self.assertEqual(mod.MODELS["default"]["model"], "legacy-work")
        self.assertEqual(mod.as_name("medium"), "default")
        self.assertEqual(mod.as_name("Medium"), "default")
        self.assertIsNone(mod.as_name("ultra"))

    def test_medium_env_prefix_fills_default(self) -> None:
        with _clean_env(MODEL_PICKER_MEDIUM_MODEL="legacy-medium"):
            mod = _load("mr_legacy_env")
        self.assertEqual(mod.MODELS["default"]["model"], "legacy-medium")

    def test_default_env_prefix_wins_over_legacy(self) -> None:
        with _clean_env(
            MODEL_PICKER_MEDIUM_MODEL="legacy-medium",
            MODEL_PICKER_DEFAULT_MODEL="canonical",
        ):
            mod = _load("mr_default_env_wins")
        self.assertEqual(mod.MODELS["default"]["model"], "canonical")


class EnvOverlay(unittest.TestCase):
    def test_named_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "low": {"model": "test-low", "provider": "test"},
                            "default": {"model": "test-default", "provider": "test"},
                            "high": {"model": "some-voice", "provider": "other"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with _clean_env(MODEL_PICKER_CONFIG=str(cfg)):
                mod = _load("mr_named_overlay")
            self.assertEqual(mod.MODELS["high"]["model"], "some-voice")
            self.assertEqual(mod.MODELS["high"]["provider"], "other")

    def test_low_model_env(self) -> None:
        with _clean_env(MODEL_PICKER_LOW_MODEL="flash-override"):
            mod = _load("mr_low_env")
        self.assertEqual(mod.MODELS["low"]["model"], "flash-override")


class RejectFourth(unittest.TestCase):
    def test_fourth_named_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "low": {"model": "a"},
                            "default": {"model": "b"},
                            "high": {"model": "c"},
                            "ultra": {"model": "d"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with _clean_env(MODEL_PICKER_CONFIG=str(cfg)):
                with self.assertRaises(Exception) as ctx:
                    _load("mr_fourth")
            self.assertIn("ultra", str(ctx.exception))


class BestFor(unittest.TestCase):
    def test_file_overlay_rebuilds_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "low": {
                                "model": "test-low",
                                "provider": "test",
                                "best_for": ["Only pings"],
                            },
                            "default": {"model": "test-default", "provider": "test"},
                            "high": {
                                "model": "test-high",
                                "provider": "test",
                                "best_for": ["Only architecture"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            with _clean_env(MODEL_PICKER_CONFIG=str(cfg)):
                mod = _load("mr_best_for_file")
        self.assertEqual(mod.MODELS["low"]["best_for"], ["Only pings"])
        self.assertIn("Only pings", mod.CLASSIFIER)
        self.assertIn("Only architecture", mod.CLASSIFIER)
        self.assertNotIn("Trivial Q&A", mod.CLASSIFIER)
        self.assertIn("Multi-step reasoning", mod.CLASSIFIER)

    def test_env_json_overlay(self) -> None:
        payload = json.dumps(["Status only"])
        with _clean_env(MODEL_PICKER_LOW_BEST_FOR=payload):
            mod = _load("mr_best_for_env")
        self.assertEqual(mod.MODELS["low"]["best_for"], ["Status only"])
        self.assertIn("Status only", mod.CLASSIFIER)


if __name__ == "__main__":
    unittest.main()
