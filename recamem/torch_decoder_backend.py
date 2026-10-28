"""Adapter for a locally supplied, hookable PyTorch causal decoder.

``GgufBackend`` cannot expose hidden-state hooks: llama.cpp is a C++ binding
with no residual-stream access. This adapter covers the other side of the
contract — a decoder the caller already has in-process as an ``nn.Module``.
It advertises hidden injection (and nothing else), exposes its layer list and
hidden size, and runs generation with whatever hooks are active.

The decoder must supply either ``model.layers`` or ``model.transformer.h``
(``nn.ModuleList`` of blocks) and ``config.hidden_size`` (or ``model.config``).
A tiny hand-written decoder is used in the tests; no weights are downloaded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from recamem.backend_contracts import (
    BackendCapabilities,
    capabilities_of,
    count_tokens,
)


class HookableDecoderBackend:
    """Wrap an in-process ``nn.Module`` decoder for scoped hidden injection."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tokenizer: Optional[Any] = None,
        context_window: int = 2048,
        output_reserve: int = 256,
        layers: Optional[Sequence[torch.nn.Module]] = None,
        hidden_size: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._context_window = int(context_window)
        self._output_reserve = int(output_reserve)
        self._layers = list(layers) if layers is not None else None
        self._hidden_size = hidden_size
        self.last_logits: Optional[torch.Tensor] = None
        self.forward_calls = 0

    # -- capabilities --------------------------------------------------- #
    def hidden_layers(self) -> List[torch.nn.Module]:
        if self._layers is None:
            self._layers = self._discover_layers()
        return self._layers

    def _discover_layers(self) -> List[torch.nn.Module]:
        candidates = [
            getattr(self.model, "layers", None),
            getattr(getattr(self.model, "transformer", None), "h", None),
            getattr(getattr(self.model, "model", None), "layers", None),
        ]
        for candidate in candidates:
            if isinstance(candidate, (torch.nn.ModuleList, list, tuple)) and len(candidate):
                return list(candidate)
        raise ValueError(
            "decoder exposes no layer list (expected model.layers or "
            "model.transformer.h)"
        )

    def hidden_size(self) -> int:
        if self._hidden_size is not None:
            return int(self._hidden_size)
        for owner in (getattr(self.model, "config", None), self.model):
            value = getattr(owner, "hidden_size", None) or getattr(owner, "n_embd", None)
            if isinstance(value, int):
                self._hidden_size = value
                return value
            if callable(value):
                self._hidden_size = int(value())
                return self._hidden_size
        raise ValueError("decoder exposes no hidden_size")

    def supports_hidden_injection(self) -> bool:
        try:
            self.hidden_layers()
            self.hidden_size()
        except Exception:
            return False
        return True

    def supports_kv_priming(self) -> bool:
        # A plain PyTorch module has no persistent cache in this adapter.
        return False

    def context_window(self) -> int:
        return self._context_window

    def output_reserve(self) -> int:
        return self._output_reserve

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=type(self).__name__,
            prompt=True,
            hidden=self.supports_hidden_injection(),
            kv_prime=False,
            context_window=self._context_window,
            output_reserve=self._output_reserve,
            tokenizer=self.tokenizer_kind,
        )

    # -- tokenizer ------------------------------------------------------ #
    @property
    def tokenizer_kind(self) -> str:
        encode = getattr(self.tokenizer, "encode", None)
        return "model" if callable(encode) else "bytes"

    def encode(self, text: str) -> List[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text))
        return [min(255, byte) for byte in text.encode("utf-8")]

    def count_tokens(self, text: str) -> int:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                return len(encode(text))
            except Exception:
                pass
        return len(text.encode("utf-8"))

    # -- generation ----------------------------------------------------- #
    def generate(self, prompt: str, max_tokens: int = 16, **_: Any) -> str:
        """Run one forward pass with the currently installed hooks."""
        tokens = self.encode(prompt)
        if not tokens:
            return ""
        ids = torch.tensor([tokens[-self._context_window:]], dtype=torch.long)
        self.forward_calls += 1
        with torch.no_grad():
            output = self.model(ids)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        self.last_logits = logits.detach()
        # Deterministic decode: map the final-token argmax to a printable char.
        index = int(torch.argmax(logits[0, -1]).item())
        return chr(65 + (index % 26)) * max(1, min(4, max_tokens))

    def prime_kv_cache(self, texts: Sequence[str]) -> Dict[str, Any]:
        from recamem.backend_contracts import UnsupportedCapabilityError

        raise UnsupportedCapabilityError("kv_cache", type(self).__name__)

    def reset_kv_cache(self) -> None:  # pragma: no cover - not advertised
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "capabilities": capabilities_of(self).to_dict(),
            "hidden_size": self.hidden_size(),
            "layers": len(self.hidden_layers()),
            "forward_calls": self.forward_calls,
        }


__all__ = ["HookableDecoderBackend"]
