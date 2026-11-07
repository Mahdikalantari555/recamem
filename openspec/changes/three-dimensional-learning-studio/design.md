# Design

## Existing state and boundaries

`companion/src/components/LearningConstellation.svelte` lazily imports Three.js,
uses a decorative constellation with a CSS fallback, caps pixel ratio, pauses
on hidden documents, respects reduced motion and disposes resources on unmount.
The scene has not been verified in a real WebGL browser, including context loss.
The existing learning API reports blocked/paused and does not train in the
background. This change augments the existing workspace, not a second frontend.

## Experience concept

Create a quiet **Learning Observatory**: a central decorative seed with optional
spatial layers representing *only* actual, authorized records (for example,
selected memory evidence or completed local jobs). The canvas is a visual aid,
not the source of truth. A semantic HTML status panel presents the current
backend state, timestamp, unavailable/stale conditions, controls and provenance.
Decorative animation must be explicitly labeled as illustrative and never imply
training activity. If no measurable data are available, show an empty or blocked
state rather than inventing nodes, progress, weights or intelligence.

Explore keyboard-accessible DOM navigation and a synchronized list/detail view
for any selectable 3D object; canvas-only interaction is insufficient. Evidence
is user data: display only what existing authorized APIs return, escape it via
framework defaults, never interpret it as code, and do not send it to external
rendering/analytics services. No network requests beyond existing relative
same-origin APIs; no scene-generated writes or tool invocations. Memory and
weight updates must be clearly distinguished, and autonomous research retains
its independent opt-in and network security gates.

## Technology and lifecycle

Start from the existing Three.js component and compare an optional Svelte-native
[Threlte](https://threlte.xyz/docs/learn/basics/render-modes/) prototype against
it before adding dependencies. Threlte's on-demand rendering is a possible fit,
not evidence it is faster in this project. Use the [Three.js manual and
examples](https://threejs.org/manual/) for scene, disposal and on-demand render
patterns; no example code/assets may be copied without a license review.
A useful inspiration for a semantic HTML-first enhancement is the
[WebGL-UI accessibility research](https://github.com/LucaArgentieri/WebGL-UI/issues/1).
The [Three.js examples](https://threejs.org/examples/) can inform interaction
and object rendering, but are not an architecture or performance guarantee.

Keep render loops independent from Svelte reactive state, share geometry and
materials where justified, and limit canvas count, draw calls and pixel ratio.
Render on demand when static; suspend work offscreen or hidden and offer an
explicit motion/3D disable control. Handle resize, setup failure, context loss,
restoration and teardown without leaks or duplicate loops. Respect reduced
motion, forced colors and contrast settings. A static 2D fallback must preserve
all operations and evidence. Do not rely on WebGL for controls, labels or state.

Define measurable budgets *before* comparing variants: initial chunk size and
load, frame/interaction responsiveness, browser CPU/GPU memory, and impact on
CPU training/inference. Evaluate low-end and target hardware and mobile; report
hardware, browser, settings, workload and methodology. GPU rendering is optional
and separate from the CPU-only ML computation. Prefer progressive degradation
and a user-visible off switch to overclaiming a universal frame rate.

## Validation and rollout

Use component tests for status mapping, freshness, provenance and semantic
controls; automated real-browser tests for keyboard, reduced motion, responsive
layout, no WebGL, context loss, cleanup and 2D fallback; manual screen reader and
contrast review. Verify idle/offscreen and contention behavior on target
hardware. Keep the present experience as rollback until gates are met. No task
that depends on the unimplemented learning supervisor or browser research may
be marked complete by visual placeholders.
