# Proposal: Three-dimensional learning studio

## Why
The user wants a creative Three.js GUI that makes local learning, memory and
progress understandable without requiring architecture or tokenizer choices.
The existing Svelte Learning Studio has an illustrative Three.js constellation,
but its WebGL behavior has not been verified in a real browser and it does not
represent live model internals. This change proposes a fuller, usable 3D
experience without presenting decoration as actual learning.

## What Changes
Extend the existing Svelte companion with an optional, local-first spatial
workspace: a calm constellation/observatory with a semantic 2D interface for
learning status, authorized sources, memory evidence and evaluation. Bind any
status or data-driven objects to measured backend records and distinguish them
from decorative elements. Provide an equivalent non-WebGL experience, motion
and accessibility preferences, resource limits and clear blocked states.
Investigate a Svelte-native Three.js wrapper only if it improves maintainability
and measured performance; do not introduce a React stack. Coordinate with
`cpu-learning-svelte` and `autonomous-incremental-learning` rather than build a
separate agent, trainer or memory store.

## Capabilities
- `surface/three-dimensional-studio`: optional 3D visualization, accessible interaction, runtime fallback and trustworthy status.

## Non-goals
No claim that model weights, learned concepts or training activity are depicted
unless a real, documented API provides that information. No network access or
background model training solely from loading a scene. No required WebGL, GPU,
VR hardware, cloud service, GGUF or new model backend. This is a proposal; the
existing decorative constellation is not the completed implementation.

## Impact
Svelte companion, Three.js scene lifecycle, same-origin status/data APIs, tests,
documentation and release evidence. Preserve all existing 2D workflows and the
CPU-only model execution path. Performance gates need measurements on actual
target hardware and must not be inferred from vendor demos.
