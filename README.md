# model-classifier — per-turn cost classification across three model tiers

A Hermes Agent plugin that classifies **each user turn** and moves the session
to the cheapest tier that can do the job, then gets out of the way when tools
fail.

Three named tiers, in ascending cost:

| Tier | Default label | For |
| --- | --- | --- |
| `low` | Quick | lookups, small edits, routine questions |
| `default` | Standard | normal development and research work |
| `high` | Expert | money, irreversible actions, security-sensitive work |

`medium` is the deprecated alias for `default` (the v0.8 lexicon) and still
resolves to that slot everywhere — config keys, env, `/medium`.

## What it does

- **Classifies on `pre_llm_call`.** One auxiliary LLM call picks the tier from
  each tier's `best_for` list. The prompt is generated from that list, so
  `config.json` is the single source of truth for routing policy.
- **Stays put unless the turn changes shape.** The classifier is biased to keep
  the previous tier; `high` never sticks across turns.
- **Compacts before switching.** A classifier-driven tier change runs the
  context handoff first, so the new model does not inherit a transcript it
  cannot afford to read.
- **Escalates on failure.** `escalate_model` (registered as a tool) moves a
  failing session up a tier after repeated tool errors, capped by
  `escalate_max`.
- **Pins on request.** `/low`, `/default`, `/high` pin a session; `/auto`
  resumes classification.
- **Repairs half-switched clients.** If a credential refresh swaps the provider
  host under a pinned session, the plugin detects the mismatch and re-applies
  the intended route instead of letting it run against the wrong endpoint.
- **Skips what it cannot classify.** Cron and subagent platforms are skipped by
  default; so is a session whose tier is pinned.

It switches models through `AIAgent.switch_model` and registers hooks only: no
Hermes or WebUI core file is edited, and the plugin never writes to SOUL.md or
MEMORY.md.

## Install

```bash
hermes plugins install <owner>/hermes-model-classifier
hermes plugins enable model-classifier
```

Manual install: copy this directory into `~/.hermes/plugins/model-classifier`,
add `model-classifier` to `plugins.enabled`, restart.

## Configuration

### Model ids (`config.json`)

The plugin ships labels and `best_for` in `config.default.json`, and **no model
ids** — ids belong to the deployment. Provide them in a `config.json` next to
the plugin:

```json
{
  "models": {
    "low":     { "model": "cheap-model",   "provider": "p1", "label": "Quick",    "best_for": ["lookups", "small edits"] },
    "default": { "model": "standard-model","provider": "p2", "label": "Standard", "best_for": ["development", "research"] },
    "high":    { "model": "expert-model",  "provider": "p2", "label": "Expert",   "best_for": ["money", "irreversible"] }
  },
  "escalate_max": "high",
  "escalation_errors": { "low": 4, "default": 3 },
  "skip_platforms": ["cron", "subagent"],
  "classifier_timeout_s": 8.0
}
```

Resolution order: `config.default.json` → `config.json` →
`$MODEL_CLASSIFIER_CONFIG` → `MODEL_CLASSIFIER_*` env → **the host's own primary
model** (`model.default` in the Hermes config) for every tier that is still
unset. An install with no plugin config therefore routes on the model it
already runs, and the plugin reports itself inert (registered, declining to
classify) rather than failing to load if the host has no primary model either.

### Environment

Canonical names are `MODEL_CLASSIFIER_*`. The pre-rename `MODEL_ROUTER_*` names
are still accepted as legacy aliases, so an existing deployment keeps its
configuration; the canonical name wins when both are set.

| Variable | Meaning |
| --- | --- |
| `MODEL_CLASSIFIER_CONFIG` | Path to a config.json (legacy: `MODEL_ROUTER_CONFIG`). |
| `MODEL_CLASSIFIER_LOW_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `low`. |
| `MODEL_CLASSIFIER_DEFAULT_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `default`. |
| `MODEL_CLASSIFIER_HIGH_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `high`. |
| `MODEL_CLASSIFIER_CLASSIFIER_TIMEOUT_S` | Per-call timeout for the classifier (default `8`, minimum `1`). |

`MODEL_CLASSIFIER_MEDIUM_*` is the deprecated alias for the `default` slot.

## Behaviour worth knowing

- **Fail-open.** Classifier timeout, malformed reply, or a dead auxiliary model
  degrades to the `low` tier for that turn — never to a failed turn. The
  timeout is deliberately short because the classifier runs inside the
  `pre_llm_call` budget.
- **`high` is rare.** The generated prompt restricts `high` to money,
  irreversible actions and security-sensitive work, because a mis-set `high` is
  the expensive failure mode.
- **Labels are cosmetic.** `Quick`/`Standard`/`Expert` are display names; the
  routing logic uses slot names only.

## Tests

```bash
python -m pytest tests/ -q
```

Pure unit tests: stock CPython, no Hermes runtime, no API calls. They cover
config precedence (catalog → file → env), the primary-model fallback, canonical
vs legacy env precedence, tier name/alias resolution, the classifier prompt
generation, escalation accounting, and the pin/reset command paths.

## Credits

[open-world-project/model-router](https://github.com/open-world-project/model-router)
for the cheap/work/voice idea. This is an independent implementation: hooks,
config schema, classifier prompt and escalation logic are all our own.

## License

MIT — see [LICENSE](LICENSE).
