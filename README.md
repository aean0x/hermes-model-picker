# model-picker — per-turn cost classification across three model tiers

A Hermes Agent plugin that classifies **each user turn**, moves the session to
the cheapest tier that can do the job, and gets out of the way when tools fail.

Three named tiers, in ascending cost:

| Tier | Default label | For |
| --- | --- | --- |
| `low` | Quick | lookups, small edits, routine questions |
| `default` | Standard | normal development and research work |
| `high` | Expert | money, irreversible actions, security-sensitive work |

`medium` is the deprecated alias for `default` (the v0.8 lexicon) and still
resolves to that slot everywhere — config keys, env, `/medium`.

## What it does

- **Classifies on `pre_llm_call`.** One bounded auxiliary call picks the tier
  from each tier's `best_for` list. The prompt is generated from that list, so
  `config.json` is the single source of truth for routing policy.
- **Applies the tier on `llm_request`.** A request middleware rewrites the
  outgoing request's `model` immediately before the provider call. Nothing else
  is touched — no client rebuild, no `AIAgent` attribute writes, no method
  wrapping, no WebUI script.
- **Stays put unless the turn changes shape.** The classifier is told the
  previous tier and is biased to keep it; `high` never sticks across turns.
- **Compacts before switching.** A classifier-driven tier change asks the
  context engine to compact at the next turn boundary, so the new model does not
  inherit a transcript it cannot afford to read.
- **Escalates on failure.** Repeated tool errors stage a checkpoint; the working
  model may then call `escalate_model` to climb one tier, capped by
  `escalate_max`.
- **Pins on request.** `/low`, `/default`, `/high` pin a session; `/auto`
  resumes classification. A pin sent through the composer (the WebUI sends
  `/high` as a message) is honoured too.
- **Skips what it cannot classify.** Cron and subagent platforms are skipped by
  default; so is a pinned session.

### Tier scope: one provider per session

A request middleware can rewrite the model id, not the API host — the client for
a request is built from the session's provider. **All three tiers must therefore
resolve to the session's own provider.** A tier that names a different provider
is refused and logged once, and the session's model stands for that turn. This
is the honest limit of the seam the catalog policy allows; a cross-provider
route needs a core change, not a plugin.

### The handoff engine (optional)

`escalate_model` and the pre-switch compaction need a context engine. The plugin
registers one only when the host config asks for it:

```yaml
context:
  engine: model-picker
```

Without that, the plugin still routes and pins; escalation changes the model and
says so, and the transcript is untouched.

### No Hermes or WebUI core file is edited

Every extension point is a documented plugin surface: `pre_llm_call`,
`post_tool_call`, the `llm_request` middleware, `ctx.llm` for the classifier
call, the tool/command registry, and the single `register_context_engine` slot.
The plugin never writes to SOUL.md or MEMORY.md.

## Install

```bash
hermes plugins install <owner>/hermes-model-picker
hermes plugins enable model-picker
```

Manual install: copy this directory into `~/.hermes/plugins/model-picker`,
add `model-picker` to `plugins.enabled`, restart.

Requires Hermes `>=0.21.5` (the `llm_request` middleware and the `ctx.llm`
facade).

## Configuration

### Model ids (`config.json`)

The plugin ships labels and `best_for` in `config.default.json`, and **no model
ids** — ids belong to the deployment. Provide them in a `config.json` next to
the plugin:

```json
{
  "models": {
    "low":     { "model": "cheap-model",    "provider": "p1", "label": "Quick",    "best_for": ["lookups", "small edits"] },
    "default": { "model": "standard-model", "provider": "p1", "label": "Standard", "best_for": ["development", "research"] },
    "high":    { "model": "expert-model",   "provider": "p1", "label": "Expert",   "best_for": ["money", "irreversible"] }
  },
  "escalate_max": "high",
  "escalation_errors": { "low": 4, "default": 3 },
  "skip_platforms": ["cron", "subagent"],
  "classifier_timeout_s": 8.0
}
```

All three `provider` values must be the session's provider — see
[Tier scope](#tier-scope-one-provider-per-session).

Resolution order: `config.default.json` → `config.json` →
`$MODEL_PICKER_CONFIG` → `MODEL_PICKER_*` env → **the host's own primary model**
(`model.default` in the Hermes config) for every tier that is still unset. An
install with no plugin config therefore routes on the model it already runs, and
the plugin reports itself inert (registered, declining to classify) rather than
failing to load if the host has no primary model either.

### Classifier model (optional)

By default the classifier runs on the session's active model through the
host-owned `ctx.llm` facade. To pin it to a cheaper model, name one and allow
the override for this plugin — the plugin-LLM facade is fail-closed and refuses
an override the host has not allowed:

```yaml
plugins:
  entries:
    model-picker:
      llm:
        allow_model_override: true
        allow_provider_override: true
        allow_task_override: true
```

```json
{ "classifier_model": "cheap-model", "classifier_provider": "p1" }
```

A refused override is logged and the classifier falls back to the active model;
classification never fails the turn.

### Environment

Canonical names are `MODEL_PICKER_*`; no earlier generation is read.

| Variable | Meaning |
| --- | --- |
| `MODEL_PICKER_CONFIG` | Path to a config.json (overrides the plugin-adjacent one). |
| `MODEL_PICKER_LOW_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `low`. |
| `MODEL_PICKER_DEFAULT_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `default`. |
| `MODEL_PICKER_HIGH_MODEL` / `_PROVIDER` / `_LABEL` / `_BEST_FOR` | Per-slot override for `high`. |
| `MODEL_PICKER_CLASSIFIER_TIMEOUT_S` | Per-call timeout for the classifier (default `8`, minimum `1`). |
| `MODEL_PICKER_CLASSIFIER_MODEL` / `_PROVIDER` | Override the classifier's own model/provider. |

`MODEL_PICKER_MEDIUM_*` is the deprecated alias for the `default` slot.

## Behaviour worth knowing

- **Fail-open.** Classifier timeout, malformed reply, or a dead model degrades to
  the `low` tier for that turn — never to a failed turn. The timeout is
  deliberately short because the classifier runs inside the `pre_llm_call`
  budget.
- **`high` is rare.** The generated prompt restricts `high` to money,
  irreversible actions and security-sensitive work, because a mis-set `high` is
  the expensive failure mode.
- **Fallbacks win.** If Hermes has already put a fallback model in the request
  payload, the middleware leaves it alone.
- **Per-turn routing costs cache reuse.** A model switch invalidates the
  provider's prompt cache for that turn; pin a session (`/default`) when the
  cache matters more than the saving.
- **Labels are cosmetic.** `Quick`/`Standard`/`Expert` are display names; the
  routing logic uses slot names only.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Pure unit tests: stock CPython, no API calls. They cover config precedence
(catalog → file → env), the primary-model fallback, canonical vs legacy env
precedence, tier name/alias resolution, the classifier prompt generation, the
request middleware (including the foreign-provider refusal and the fallback
case), the registered surface against `plugin.yaml`, escalation accounting, the
pin/reset command paths, and the handoff engine (tail preservation, turn
boundary, one-shot semantics). The engine tests use the real `ContextCompressor`
when the Hermes runtime is importable and a stubbed base otherwise; the subclass
logic under test is the same either way.

## Migrating from 0.11.x

- The model switch moved from `AIAgent` surgery to the `llm_request`
  middleware. Nothing in the plugin writes an agent attribute any more.
- `provider_hosts` config is gone with the half-switch repair. A tier on another
  provider is refused rather than half-applied.
- The WebUI extension is gone (`webui/`); the WebUI already lists plugin
  commands, and `/low`, `/default`, `/high`, `/auto` work from the composer.
- The handoff engine now registers only when `context.engine: model-picker`.
- Requires Hermes `>=0.21.5`.

## Credits

[open-world-project/model-router](https://github.com/open-world-project/model-router)
for the cheap/work/voice idea. This is an independent implementation: hooks,
config schema, classifier prompt and escalation logic are all our own.

## License

MIT — see [LICENSE](LICENSE).
