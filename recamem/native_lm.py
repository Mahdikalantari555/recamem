"""CPU CandidateLM inference for explicitly identified, locally trained checkpoints."""
from pathlib import Path
import torch
from src.model.cores.registry import build_candidate_lm
from src.model.cores.tokenizer import UTF8ByteTokenizer


class NativeLMBackend:
    tokenizer_kind = 'bytes'

    def __init__(self, checkpoint, context_window=256, max_output_bytes=128):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'Native LM checkpoint not found: {path}')
        # Only load trusted local training checkpoints: torch pickle is executable.
        obj = torch.load(path, map_location='cpu', weights_only=True)
        cfg = obj.get('meta', {}).get('config', {})
        if cfg.get('tokenizer') != 'utf8_bytes_256' or cfg.get('vocab_size') != 256 or not cfg.get('core_id'):
            raise ValueError('Checkpoint lacks supported architecture and byte tokenizer metadata')
        self.config = dict(cfg)
        self.training_progress = {'step': obj.get('step'), 'tokens_seen': obj.get('tokens_seen')}
        self.model = build_candidate_lm(cfg)
        self.model.load_state_dict(obj['model_state'], strict=True)
        self.model.eval()
        self.tokenizer = UTF8ByteTokenizer()
        self._context = int(context_window)
        self._reserve = int(max_output_bytes)
        if self._context < 1 or self._reserve < 0:
            raise ValueError('invalid context or output reserve')

    def supports_hidden_injection(self): return False
    def supports_kv_priming(self): return False
    def context_window(self): return self._context
    def output_reserve(self): return self._reserve
    def count_tokens(self, text): return len(text.encode('utf-8'))
    def tokenize(self, text): return self.tokenizer.encode(text)
    def reset(self): self.model.reset_state()

    def generate(self, prompt, max_tokens=128, temperature=0.0, **kwargs):
        if temperature < 0:
            raise ValueError('temperature must be nonnegative')
        ids = self.tokenizer.encode(prompt)[-self._context:]
        if not ids:
            raise ValueError('prompt must be nonempty')
        result = []
        with torch.no_grad():
            for _ in range(min(max(0, int(max_tokens)), self._reserve)):
                logits = self.model(torch.tensor([ids[-self._context:]], dtype=torch.long))[0, -1]
                token = int(torch.argmax(logits) if temperature == 0 else torch.multinomial(torch.softmax(logits / temperature, -1), 1).item())
                if token == self.tokenizer.eos_id:
                    break
                ids.append(token); result.append(token)
        return self.tokenizer.decode(result)
