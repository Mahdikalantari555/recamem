#!/usr/bin/env python3
"""Generate with a trusted CPU-native byte LM checkpoint (no GGUF)."""
import argparse
from recamem.native_lm import NativeLMBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    backend = NativeLMBackend(args.checkpoint)
    print(backend.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature))

if __name__ == '__main__':
    main()
