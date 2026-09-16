"""model-picker settings — three named tiers (low < default < high).

This module does not own model IDs. Slot names are low/default/high.
"medium" is the deprecated alias for "default" (the v0.8 lexicon) and is
still accepted in config keys, MODEL_CLASSIFIER_MEDIUM_* env, and
pin/classifier tokens. Labels and `best_for` default from
config.default.json. Model id and provider come from, in order:
config.default.json, config.json (Nix `hermesPnP.models`), a
MODEL_PICKER_CONFIG path, MODEL_PICKER_* env — and, if none of those
name a model, the host's own primary model (`model.default` in the Hermes
config) for all three tiers. No Hermes/WebUI core files are edited.

Env names are canonical `MODEL_PICKER_*`. Both earlier generations still work
as legacy aliases — `MODEL_CLASSIFIER_*` (the 0.10 name) and `MODEL_ROUTER_*`
(the original) — so an existing deployment keeps its configuration; the
canonical name wins when more than one is set.

The `best_for` list on each tier is the source of truth for the
classifier prompt. A short steer block (prefer-low on doubt; high is
rare) is generated with it — not a second matrix. Auto always
classifies all three tiers.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).resolve().parent

NAMES: tuple[str, ...] = ("low", "default", "high")
RANK: dict[str, int] = {"low": 0, "default": 1, "high": 2}

# Deprecated tier token. Accepted everywhere a tier name is accepted and
# always resolves to the canonical "default" slot, so a v0.8 consumer
# config (keys, MODEL_PICKER_MEDIUM_* env, /medium pin) keeps working.
LEGACY_NAMES: dict[str, str] = {"medium": "default"}
_ALIASES: dict[str, str] = {**LEGACY_NAMES, **{n: n for n in NAMES}}

# Per-slot env prefixes, lowest precedence first: the deprecated
# MODEL_ROUTER_MEDIUM_* group, then the pre-rename MODEL_ROUTER_* names, then
# the canonical MODEL_CLASSIFIER_* names. A later entry overwrites an earlier
# one, so canonical wins over legacy and an explicit DEFAULT_* wins over the
# deprecated MEDIUM_* alias.
_ENV_SLOTS: tuple[tuple[str, str], ...] = (
    ("low", "MODEL_ROUTER_LOW_"),
    ("default", "MODEL_ROUTER_MEDIUM_"),
    ("default", "MODEL_ROUTER_DEFAULT_"),
    ("high", "MODEL_ROUTER_HIGH_"),
    ("low", "MODEL_CLASSIFIER_LOW_"),
    ("default", "MODEL_CLASSIFIER_MEDIUM_"),
    ("default", "MODEL_CLASSIFIER_DEFAULT_"),
    ("high", "MODEL_CLASSIFIER_HIGH_"),
    ("low", "MODEL_PICKER_LOW_"),
    ("default", "MODEL_PICKER_MEDIUM_"),
    ("default", "MODEL_PICKER_DEFAULT_"),
    ("high", "MODEL_PICKER_HIGH_"),
)

# Knobs read outside the per-slot loop: canonical name first, then the two
# legacy generations (model-classifier, then the original model-router).
_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "CONFIG": ("MODEL_PICKER_CONFIG", "MODEL_CLASSIFIER_CONFIG", "MODEL_ROUTER_CONFIG"),
    "CLASSIFIER_TIMEOUT_S": (
        "MODEL_PICKER_CLASSIFIER_TIMEOUT_S",
        "MODEL_CLASSIFIER_CLASSIFIER_TIMEOUT_S",
        "MODEL_ROUTER_CLASSIFIER_TIMEOUT_S",
    ),
}


def env_knob(name: str) -> str | None:
    """First non-empty value among a knob's env names (canonical first)."""
    for key in _ENV_ALIASES.get(
        name,
        (
            f"MODEL_PICKER_{name}",
            f"MODEL_CLASSIFIER_{name}",
            f"MODEL_ROUTER_{name}",
        ),
    ):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw)
    return None

_CATALOG_PATH = _PLUGIN_DIR / "config.default.json"


def _slot_shells() -> dict[str, dict[str, Any]]:
    """Three named slots with no model IDs. Catalog / config / env fill them."""
    return {
        name: {
            "label": name.capitalize(),
            "short": name.capitalize(),
            "model": "",
            "provider": "",
            "best_for": [],
        }
        for name in NAMES
    }


def _missing_ids(models: dict[str, dict[str, Any]]) -> list[str]:
    """Slot names with no model id."""
    return [
        name
        for name in NAMES
        if not str(models.get(name, {}).get("model") or "").strip()
    ]


def _hermes_primary() -> tuple[str, str]:
    """(model, provider) of the host's primary model, or ("", "").

    An install with no config.json and no env must still work: the three tiers
    then fall back to whatever model Hermes itself runs on. This also keeps
    the catalog's isolated `register()` probe honest, where no plugin config
    exists at all.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
    except Exception:
        return "", ""
    if not isinstance(cfg, dict):
        return "", ""
    raw = cfg.get("model")
    if isinstance(raw, dict):
        name = raw.get("default") or raw.get("name") or raw.get("model")
        provider = raw.get("provider")
    else:
        name = raw
        provider = cfg.get("provider")
    if not isinstance(name, str):
        name = ""
    if not isinstance(provider, str):
        provider = ""
    return name.strip(), provider.strip()

# provider -> host heuristics for half-switch repair (model set, old API host).
DEFAULT_PROVIDER_HOSTS: dict[str, dict[str, list[str]]] = {
    "deepseek": {"forbid": ["x.ai", "xai"], "prefer": ["deepseek"]},
    "deepseek-chat": {"forbid": ["x.ai", "xai"], "prefer": ["deepseek"]},
    "xai": {"forbid": ["deepseek.com"], "prefer": ["x.ai", "xai"]},
    "xai-oauth": {"forbid": ["deepseek.com"], "prefer": ["x.ai", "xai"]},
    "x-ai": {"forbid": ["deepseek.com"], "prefer": ["x.ai", "xai"]},
}


class SettingsError(ValueError):
    """Invalid model-picker configuration."""


def as_name(raw: Any) -> str | None:
    """Map a config/env/classifier token onto low|default|high, or None.

    "medium" resolves to "default" (deprecated alias).
    """
    if raw is None:
        return None
    return _ALIASES.get(str(raw).strip().lower())


def as_best_for(raw: Any) -> list[str]:
    """Coerce a config/env best_for value to a list of descriptors."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                return [text]
        else:
            return [part.strip() for part in text.split(";") if part.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _coerce_models_map(raw: Any, *, origin: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    extra: list[str] = []
    for key, meta in raw.items():
        name = as_name(key)
        if name is None:
            extra.append(str(key))
            continue
        if isinstance(meta, dict):
            out[name] = dict(meta)
    if extra:
        raise SettingsError(
            f"model-picker: {origin} declares unknown models {extra}; "
            "only low, default, high are allowed"
        )
    if len(raw) > 3 or len(out) > 3:
        raise SettingsError(
            f"model-picker: {origin} declares {len(raw)} models; exactly 3 are allowed"
        )
    return out


def _deep_merge(
    base: dict[str, dict[str, Any]], overlay: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    merged = deepcopy(base)
    for name, meta in overlay.items():
        if name in merged:
            merged[name] = {**merged[name], **meta}
        else:
            merged[name] = dict(meta)
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _generated_classifier(models: dict[str, dict[str, Any]]) -> str:
    """Build the turn-start triage prompt from `best_for` plus a steer block.

    Keys stay low/default/high. Labels (Quick/Standard/Expert) are display.
    """
    names = NAMES
    lines = ["Route this turn to the cheapest model that will do it well.", ""]
    for name in names:
        if name not in models:
            continue
        meta = models[name]
        label = meta.get("label") or name.capitalize()
        best = meta.get("best_for") or []
        extra = "; ".join(str(x) for x in best) if best else label
        lines.append(f"{name} = {label} — {extra}")
    lines.append("")
    lines.append("high is ONLY the cases listed above — it is rare.")
    lines.append(
        "When uncertain between low and default, prefer low — a wrong low "
        "route is cheaply corrected by escalation."
    )
    lines.append(
        "The previous turn's tier is given in the request. If it was low or "
        "default, prefer to keep it unless the scope or topic of the work has "
        "significantly changed. If it was high, do not carry it over — "
        "classify at-will."
    )
    lines.append("Respond with ONLY one word: " + " or ".join(names) + ".")
    return "\n".join(lines)


def _apply_file(data: dict[str, Any], state: dict[str, Any], *, origin: str) -> None:
    if "models" in data:
        state["models"] = _deep_merge(
            state["models"], _coerce_models_map(data["models"], origin=f"{origin}.models")
        )
    if "provider_hosts" in data and isinstance(data["provider_hosts"], dict):
        for prov, spec in data["provider_hosts"].items():
            if isinstance(spec, dict):
                state["provider_hosts"][str(prov)] = {
                    "forbid": list(spec.get("forbid") or []),
                    "prefer": list(spec.get("prefer") or []),
                }
    if "escalate_max" in data:
        name = as_name(data["escalate_max"])
        if name:
            state["escalate_max"] = name
    if "escalation_errors" in data and isinstance(data["escalation_errors"], dict):
        errors: dict[str, int] = {}
        extra: list[str] = []
        for key, val in data["escalation_errors"].items():
            name = as_name(key)
            if name is None:
                extra.append(str(key))
                continue
            errors[name] = int(val)
        if extra:
            raise SettingsError(
                f"model-picker: {origin}.escalation_errors has unknown keys {extra}; "
                "only low, default, high are allowed"
            )
        state["escalation_errors"] = errors
    if "skip_platforms" in data and isinstance(data["skip_platforms"], list):
        state["skip_platforms"] = [str(x) for x in data["skip_platforms"]]
    if data.get("classifier_system"):
        state["classifier_system"] = str(data["classifier_system"])
    if "handoff_tail_chars" in data:
        state["handoff_tail_chars"] = max(1000, int(data["handoff_tail_chars"]))
    if "classifier_context_chars" in data:
        state["classifier_context_chars"] = max(1000, int(data["classifier_context_chars"]))
    if "classifier_timeout_s" in data:
        state["classifier_timeout_s"] = max(1.0, float(data["classifier_timeout_s"]))


def load_settings() -> dict[str, Any]:
    state: dict[str, Any] = {
        "models": _slot_shells(),
        "provider_hosts": deepcopy(DEFAULT_PROVIDER_HOSTS),
        "escalate_max": "high",
        "escalation_errors": {"low": 4, "default": 3},
        "skip_platforms": ["cron", "subagent"],
        "classifier_system": None,
        "handoff_tail_chars": 64000,
        "classifier_context_chars": 12000,
        # Bounded per-call timeout for the auxiliary tier classifier.  The
        # classifier runs synchronously inside the pre_llm_call hook, whose
        # gateway budget is 30s; an unbounded LLM round-trip there (observed
        # 50-80s during API latency spikes) blew the budget, got the hook
        # abandoned mid-run, and cascaded into "skipped after previous
        # timeout" storms.  Fail-open to low is the designed degradation, so
        # a short cap only costs a classification, never the turn.
        "classifier_timeout_s": 8.0,
    }

    config_env = env_knob("CONFIG")
    for candidate, origin in (
        (_CATALOG_PATH, "config.default.json"),
        (_PLUGIN_DIR / "config.json", "config.json"),
        (Path(config_env) if config_env else None, "MODEL_PICKER_CONFIG"),
    ):
        if candidate is None:
            continue
        data = _load_json(candidate)
        if data:
            _apply_file(data, state, origin=origin)

    models = state["models"]
    extra = [name for name in models if name not in RANK]
    if extra:
        raise SettingsError(
            f"model-picker: unknown models {extra}; only low, default, high are allowed"
        )
    if len(models) > 3:
        raise SettingsError("model-picker: a fourth model is not allowed")

    for name, prefix in _ENV_SLOTS:
        model = os.environ.get(prefix + "MODEL")
        provider = os.environ.get(prefix + "PROVIDER")
        label = os.environ.get(prefix + "LABEL")
        if model:
            models[name]["model"] = model.strip()
        if provider:
            models[name]["provider"] = provider.strip()
        if label:
            models[name]["label"] = label.strip()
            models[name].setdefault("short", label.strip().split()[0])
        raw_best = os.environ.get(prefix + "BEST_FOR")
        if raw_best and raw_best.strip():
            models[name]["best_for"] = as_best_for(raw_best)

    # Nothing named a model id. Default to the host's own primary model for all
    # three tiers: a stock install (no config.json, no env) then routes, and the
    # catalog's isolated register() probe — where no plugin config exists at all
    # — still loads and registers its declared surface. Refusing to import here
    # would be a load failure, not a configuration hint.
    missing = _missing_ids(models)
    if missing:
        primary_model, primary_provider = _hermes_primary()
        if primary_model:
            for name in missing:
                models[name]["model"] = primary_model
                if primary_provider and not str(models[name].get("provider") or "").strip():
                    models[name]["provider"] = primary_provider
    state["unconfigured"] = _missing_ids(models)
    state["configured"] = not state["unconfigured"]

    env_timeout = env_knob("CLASSIFIER_TIMEOUT_S")
    if env_timeout:
        try:
            state["classifier_timeout_s"] = max(1.0, float(env_timeout))
        except (TypeError, ValueError):
            pass

    if state["escalate_max"] not in models:
        state["escalate_max"] = "high" if "high" in models else next(iter(models))

    for name, meta in models.items():
        meta.setdefault("short", name.capitalize())
        meta.setdefault("label", name.capitalize())
        meta["best_for"] = as_best_for(meta.get("best_for"))

    if not state["classifier_system"]:
        state["classifier_system"] = _generated_classifier(models)

    return state


_SETTINGS = load_settings()

MODELS: dict[str, dict[str, Any]] = _SETTINGS["models"]
ESCALATE_MAX: str = _SETTINGS["escalate_max"]
ESCALATION_ERRORS: dict[str, int] = _SETTINGS["escalation_errors"]
SKIP_PLATFORMS: frozenset[str] = frozenset(_SETTINGS["skip_platforms"])
PROVIDER_HOSTS: dict[str, dict[str, list[str]]] = _SETTINGS["provider_hosts"]
CLASSIFIER: str = _SETTINGS["classifier_system"]
HANDOFF_TAIL_CHARS: int = _SETTINGS["handoff_tail_chars"]
CLASSIFIER_CONTEXT_CHARS: int = _SETTINGS["classifier_context_chars"]
CLASSIFIER_TIMEOUT_S: float = _SETTINGS["classifier_timeout_s"]
# False when no tier has a model id and the host has no primary model either.
# Everything stays registered; the router just declines to classify.
CONFIGURED: bool = bool(_SETTINGS["configured"])
UNCONFIGURED_SLOTS: tuple[str, ...] = tuple(_SETTINGS["unconfigured"])


def webui_models() -> list[dict[str, str]]:
    out = []
    for name in NAMES:
        if name not in MODELS:
            continue
        meta = MODELS[name]
        label = str(meta.get("label") or name.capitalize())
        out.append(
            {
                "cmd": f"/{name}",
                "label": label,
                "short": str(meta.get("short") or name.capitalize()),
                "model": str(meta.get("model") or ""),
                "title": f"Pin {label}",
            }
        )
    out.append(
        {
            "cmd": "/auto",
            "label": "Auto",
            "short": "Auto",
            "model": "",
            "title": "Resume per-turn routing",
        }
    )
    return out
