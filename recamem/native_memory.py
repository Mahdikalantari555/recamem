"""Explicit experimental bridge from existing memory to native generation.

Retrieved text is untrusted data: prompt formatting is not an injection defense.
No tools or writes are exposed here. This adapter is not the default chat path,
not a claim of factual grounding, and does not change model weights.
"""
from __future__ import annotations

from typing import Any

from recamem.backend_contracts import fit_prompt
from recamem.engine import NO_INFORMATION_MESSAGE


class NativeMemorySession:
    """Pass budgeted existing recall evidence to an explicitly chosen backend."""

    def __init__(self, engine: Any, backend: Any):
        self.engine = engine
        self.backend = backend

    def answer(self, query: str, *, top_k: int = 3, max_tokens: int = 32) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be non-empty')
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            raise ValueError('top_k must be an integer from 1 to 10')
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 64:
            raise ValueError('max_tokens must be an integer from 1 to 64')
        reserve = self.backend.output_reserve()
        if max_tokens > reserve:
            raise ValueError('max_tokens exceeds the backend output reserve')
        hits = self.engine.recall(query, top_k=top_k)
        if not isinstance(hits, list):
            raise ValueError('recall must return a list of evidence records')
        # Filter non-text evidence before fitting so the metadata below stays
        # aligned with the exact facts passed to the backend.
        facts = [hit for hit in hits if isinstance(hit, dict) and isinstance(hit.get('text'), str) and hit['text'].strip()]
        plan = fit_prompt(query, facts, backend=self.backend, output_reserve=max_tokens)
        included = facts[:plan.fact_count]
        result = {
            'response': NO_INFORMATION_MESSAGE,
            'status': 'no_information', 'backend': 'native_pytorch',
            'memory_used': False, 'model_weights_updated': False,
            'evidence': [], 'prompt_budget': plan.to_dict(),
            'warning': 'Experimental generation may be inaccurate; recalled text is untrusted and not verified.',
        }
        # Preserve the engine's no-evidence -> no-sampling invariant, including
        # cases where all evidence was too large for the prompt budget.
        if not included:
            return result
        result['evidence'] = [
            {key: hit[key] for key in ('id', 'text', 'source', 'origin', 'source_url',
                                       'source_query', 'timestamp', 'similarity', 'channels',
                                       'retrieval_store', 'source_cluster_ids') if key in hit}
            for hit in included
        ]
        result['response'] = self.backend.generate(plan.prompt, max_tokens=max_tokens)
        result['status'] = 'generated'
        result['memory_used'] = True
        return result
