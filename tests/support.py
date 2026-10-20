"""Shared test helpers.

The engine subclasses Hermes's ``ContextCompressor``. On a host without the
Hermes runtime (no ``openai``/``pydantic``/``ruamel``) that base class cannot be
imported, so ``install_engine_stub`` swaps in a base that implements only the
surface the subclass inherits. The subclass logic under test — tail
preservation, the turn-boundary trim, the router seam — is the real code either
way; only the inherited base is fake. Where the real base imports, nothing is
stubbed and the integration is exercised for real.
"""

from __future__ import annotations

import sys
import types

# Marked on the stub so a test can tell which base it ran against.
STUB_FLAG = "__model_picker_test_stub__"


def install_engine_stub() -> bool:
    """Ensure ``agent.context_compressor`` imports. True = real base, False = stub."""
    try:
        import agent.context_compressor  # noqa: F401

        return True
    except Exception:
        pass

    package = sys.modules.get("agent") or types.ModuleType("agent")
    stub = types.ModuleType("agent.context_compressor")

    class ContextCompressor:
        def __init__(self, *args, model: str = "", **kwargs):
            self.model = model

        def on_session_start(self, session_id, **kwargs):
            return None

        def should_compress_info(self, prompt_tokens=None):
            return False, None

        def select_context(self, request_messages, **kwargs):
            return None

    stub.ContextCompressor = ContextCompressor  # type: ignore[attr-defined]
    setattr(stub, STUB_FLAG, True)
    sys.modules["agent"] = package
    sys.modules["agent.context_compressor"] = stub
    return False


def engine_base_is_stubbed() -> bool:
    module = sys.modules.get("agent.context_compressor")
    return bool(getattr(module, STUB_FLAG, False))
