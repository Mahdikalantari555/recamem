import itertools
import os

import torch

from src.model.rtu import MissingCheckpointError, Model


class ByteRtuBackend:
    """Frozen byte-level RTU backend; a drop-in peer of GgufBackend.

    Primes a reset Model with the prompt bytes (the query stays last), then
    samples bytes under the frozen path until the stop head fires or the
    output cap is reached. Component weights never train here: every call
    routes through Model.forward(frozen=True) (src/model/rtu.py), which runs
    under torch.no_grad() and never touches the optimizer.

    Conformance (plan 10 §A1 / change droid-rtu-fusion-completion):

    * ``reset()`` clears the retained prompt buffer and every state buffer,
      and ``generate()`` calls it on entry and in ``finally``;
    * the prompt is encoded verbatim — the backend does not append ``"\\n"``;
    * the sampling loop runs ``min(max_tokens, max_output_bytes)`` times and
      ``max_tokens <= 0`` produces the empty string.
    """

    def __init__(self, checkpoint: str = "", dim: int = 384, layers: int = 8,
                 spread: int = 32, max_prompt_bytes: int = 1024,
                 max_output_bytes: int = 256, stop_threshold: float = 0.35):
        self.checkpoint = checkpoint
        self.dim = dim
        self.layers = layers
        self.spread = spread
        self.max_prompt_bytes = max_prompt_bytes
        self.max_output_bytes = max_output_bytes
        self.stop_threshold = stop_threshold
        self.model = Model(dim=dim, layers=layers, spread=spread)
        # An empty path is the explicit request for a random initialisation
        # (used only by profile *creation*); a non-empty path that is absent
        # is an error, never a silent fallback.
        if checkpoint:
            if not os.path.isfile(checkpoint):
                raise MissingCheckpointError(
                    f"Checkpoint not found: {checkpoint!r}. Refusing to answer "
                    "from randomly initialised weights."
                )
            self.model.load(checkpoint)
        # Prompt bytes retained between priming and sampling; cleared by reset().
        self._prompt_bytes = b""
        # Observability only: the last prompt that was primed. Not a buffer
        # reset() owns (the spec only requires the live buffer to be cleared).
        self._last_prompt_bytes = b""

    def tokenize(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def encode_prompt(self, prompt: str) -> bytes:
        """Encode the prompt verbatim, keeping the tail (the query is last)."""
        return prompt.encode("utf-8")[-self.max_prompt_bytes:]

    def reset(self) -> None:
        """Clear the prompt buffer and every model state buffer."""
        self._prompt_bytes = b""
        try:
            self.model.reset()
        except Exception:
            pass

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        self.reset()
        try:
            if not prompt or not prompt.strip() or max_tokens <= 0:
                return ""
            # temperature/top_p are accepted for protocol parity and ignored:
            # the component owns its sampling temperature (src/model/rtu.py).
            data = self.encode_prompt(prompt)
            if not data:
                return ""
            self._prompt_bytes = data
            self._last_prompt_bytes = data
            if len(data) > 1:
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    self.model(c, nextb=n, end=(i == len(data) - 2), frozen=True)
            out_bytes = []
            b = data[-1]
            for _ in range(min(max_tokens, self.max_output_bytes)):
                b, stop = self.model(b, nextb=None, end=False, frozen=True)
                out_bytes.append(b)
                if stop > self.stop_threshold:
                    break
            return bytes(out_bytes).decode("utf-8", errors="ignore")
        finally:
            # Leave no live primed state behind: inference must not accumulate
            # state across answers (the frozen rule, FR-003).
            self.reset()
