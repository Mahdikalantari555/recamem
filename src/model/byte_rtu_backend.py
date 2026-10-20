import itertools

import torch

from src.model.rtu import Model


class ByteRtuBackend:
    """Frozen byte-level RTU backend; a drop-in peer of GgufBackend.

    Primes a reset Model with the prompt bytes (the query stays last), then
    samples bytes under the frozen path until the stop head fires or the byte
    cap is reached. Component weights never train here: every call routes
    through Model.forward(frozen=True) (src/model/rtu.py:261-265), which runs
    under torch.no_grad() and never touches the optimizer.
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
        # No-op when the path is absent (src/model/rtu.py:355-357); callers that
        # declare a checkpoint must verify the file themselves.
        self.model.load(checkpoint)

    def tokenize(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        if not prompt or not prompt.strip():
            return ""
        # Keep the prompt tail: the query owns the final state (the recurrence
        # is an EMA whose tail dominates), and the byte cap bounds priming cost.
        # max_tokens/temperature/top_p are accepted for protocol parity and
        # ignored; the component owns its sampling temperature
        # (src/model/rtu.py:228-233) and its own byte cap.
        data = (prompt + "\n").encode("utf-8")[-self.max_prompt_bytes:]
        self.model.reset()
        try:
            if len(data) > 1:
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    self.model(c, nextb=n, end=(i == len(data) - 2), frozen=True)
            out_bytes = []
            b = data[-1]
            for _ in range(self.max_output_bytes):
                b, stop = self.model(b, nextb=None, end=False, frozen=True)
                out_bytes.append(b)
                if stop > self.stop_threshold:
                    break
            return bytes(out_bytes).decode("utf-8", errors="ignore")
        finally:
            # Leave no live primed state behind: inference must not accumulate
            # state across answers (the frozen rule, FR-003).
            self.model.reset()
