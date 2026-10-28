# Recamem --- UI/UX Design Specification

**Document:** `design.md`\
**Status:** Initial design proposal\
**Product type:** Local-first desktop application for neural-model
research and experimentation

## 1. Product vision

Recamem is a desktop research workbench for developing, running,
inspecting, and evaluating compact neural architectures. Its interface
should feel like a focused neural-systems laboratory rather than a
generic chatbot or an enterprise analytics dashboard.

The UI must make model behavior inspectable: users should be able to
distinguish text generation, classification, routing, retrieval, and
abstention; inspect the evidence and confidence behind a decision; and
compare experiments without confusing illustrative data with measured
results.

The interface is a client for the model runtime, not part of the neural
architecture itself.

## 2. Design principles

1.  **Local-first:** make it obvious when the model is running locally,
    which device is being used, and where model files and experiment
    results are stored.
2.  **Research-oriented:** prioritize reproducibility, inspectability,
    and comparisons over decorative visualizations.
3.  **Evidence before claims:** show measured metrics only when the
    backend provides them. Clearly label unavailable, estimated, or
    illustrative values.
4.  **Progressive complexity:** the Playground should be approachable;
    architecture internals and experiment controls can expose deeper
    detail.
5.  **Fast and lightweight:** avoid heavy animations, oversized
    dashboards, and unnecessary background work.
6.  **Separation of concerns:** UI, runtime API, model implementation,
    and experiment storage must remain separate.
7.  **Accessible by default:** support keyboard navigation, readable
    contrast, scalable text, and clear focus states.

## 3. Target platforms and technology

### Recommended stack

-   **Desktop shell:** Tauri 2
-   **UI:** React + TypeScript
-   **Build tooling:** Vite
-   **Styling:** Tailwind CSS
-   **Components:** shadcn/ui, customized to match Recamem's visual
    identity
-   **Charts:** Apache ECharts, added only where real time-series or
    comparison data is available
-   **Icons:** Lucide
-   **Backend:** existing PyTorch runtime exposed through a local API
-   **API transport:** HTTP on loopback for the initial implementation;
    WebSocket or server-sent events for streaming and live progress when
    needed

Keep the model runtime in Python initially. Tauri/Rust should own the
desktop shell and desktop integration, not duplicate model logic.

Do not introduce a database, state-management library, or charting
package unless a real requirement justifies it. Use React state for
local UI state and a small API client layer for backend communication.

## 4. Visual identity

### Overall direction

Use a dark, technical interface with restrained lime accents. It should
resemble a precise developer tool, not a cyberpunk game UI.

-   **Primary background:** near-black charcoal
-   **Panel background:** slightly lighter charcoal
-   **Borders:** subtle neutral gray
-   **Primary text:** soft white
-   **Secondary text:** cool gray
-   **Accent:** muted electric lime
-   **Status colors:** conventional green, amber, and red, used only for
    meaningful states

Suggested starting tokens:

  Token              Value       Usage
  ------------------ ----------- -------------------------------------
  `background`       `#111318`   Main application background
  `surface`          `#1D2027`   Cards and side panels
  `surface-raised`   `#252933`   Menus and elevated panels
  `border`           `#343943`   Dividers and outlines
  `text-primary`     `#F3F4F6`   Main content
  `text-secondary`   `#9CA3AF`   Labels and metadata
  `accent`           `#B5F36D`   Active state and primary highlights
  `success`          `#8ED081`   Successful operations
  `warning`          `#F0C674`   Warnings
  `danger`           `#F07878`   Errors and destructive actions

Treat these as design tokens, not hard-coded values in individual
components. Ensure accent text and controls maintain accessible
contrast. Do not use lime for large areas or every interactive element.

### Typography

-   Use a clean sans-serif UI font such as Inter or the platform
    default.
-   Use a monospace font for tensor shapes, logs, model IDs, paths,
    metrics, and code.
-   Use compact but readable labels; do not make body text tiny to
    imitate a terminal.
-   Prefer clear hierarchy over large marketing-style headings.

### Shape and spacing

-   Use an 8-point spacing rhythm.
-   Use moderate corner radii, typically 8--12 px.
-   Keep borders subtle and avoid excessive shadows.
-   Prefer stable layouts and aligned columns over floating decorative
    cards.
-   Avoid gradients, glassmorphism, animated backgrounds, and ornamental
    neural-network graphics.

## 5. Application information architecture

The main navigation should contain:

1.  **Playground** --- interact with the loaded model.
2.  **Architecture Lab** --- inspect model configuration and
    architecture metadata.
3.  **Training** --- monitor training jobs, if supported by the backend.
4.  **Benchmarks** --- compare model versions and baselines.
5.  **Memory** --- inspect memory state and retrieval records, only if
    implemented.
6.  **Experiments** --- review configurations, runs, artifacts, and
    reproducibility metadata.
7.  **Settings** --- runtime, storage, appearance, and diagnostics.

Hide or mark unsupported pages as unavailable rather than showing fake
functionality. For the first release, implement Playground,
model/runtime status, and a minimal Experiments view. Other pages can
follow as backend capabilities mature.

## 6. Main application shell

### Layout

-   **Left navigation rail:** product mark, navigation links, current
    model, and runtime status.
-   **Main workspace:** page-specific content.
-   **Optional contextual panel:** decision trace, model details, or
    experiment metadata.
-   **Bottom status area:** runtime connection, device, active job, and
    error indicator.

The navigation rail should be collapsible. The main content should work
at common laptop resolutions and avoid requiring horizontal scrolling
for ordinary workflows.

### Global status

Display only information confirmed by the backend:

-   Runtime connection: connected, disconnected, or starting.
-   Loaded model name and revision.
-   Device: CPU or other detected device.
-   Active operation: idle, generating, training, evaluating, or
    loading.
-   Model storage location, where useful.

Do not infer that the model is using a GPU or that a job is running
merely because a page is open.

## 7. Playground

The Playground is the primary user-facing page.

### Layout

**Header** - Current model and revision - Runtime state - Controls for
generation settings when supported

**Conversation/work area** - Input editor - Submit/run action - Output
region - Cancel action while an operation is active - Clear distinction
between user input, model output, tool results, and system diagnostics

**Decision inspector** - Selected action: generate, classify, retrieve,
tool call, or abstain - Class scores or probabilities when supplied by
the model - Confidence and calibration metadata, if available - Relevant
memory/retrieval references - Tool arguments and execution status, when
applicable - Latency and token counts, if measured

The inspector must not fabricate explanations or confidence scores. If
the model does not expose internal routing information, show "Decision
trace unavailable" rather than inventing one.

### Interaction modes

Provide a task selector only for capabilities actually supported by the
runtime:

-   Generate text
-   Classify
-   Choose among options
-   Route an action
-   Evaluate a prompt

The selected mode should configure the request schema. Do not imply that
the model can perform a task simply because the UI offers a tab for it.

### Streaming

If token streaming is implemented: - Append output incrementally. - Keep
the interface responsive. - Provide a clear stop control. - Show elapsed
time and generated-token count only when the runtime reports them. -
Handle cancellation and partial output without losing the prompt.

### Empty and error states

The initial empty state should explain how to load a model and run a
first request. Errors should identify the failed operation and provide
actionable details, with raw diagnostics available in a collapsible
section.

## 8. Architecture Lab

The Architecture Lab presents implementation metadata, not a speculative
visualization.

Potential sections:

-   Model configuration and revision
-   Parameter count
-   Component list and dimensions
-   Input/output tensor shapes
-   Sequence length and state dimensions
-   Memory components
-   Generation pathway
-   Classification/routing pathway
-   Trainable versus frozen parameters
-   Configuration file and checkpoint references

Where the backend can provide a real computation graph, show it as an
interactive node-and-edge diagram. Otherwise, use a documented static
architecture diagram or a structured component table. Clearly label
conceptual diagrams as conceptual.

Do not imply that a diagram is the exact runtime graph unless it is
generated from the actual implementation.

## 9. Training

Training controls and telemetry must reflect the existing backend's real
capabilities.

Possible controls: - Dataset path or dataset selection - Tokenizer and
vocabulary configuration - Sequence length - Batch size or effective
batch size - Learning rate and optimizer - Training steps or epochs -
Seed - Checkpoint destination

Show configuration validation before launching a run. Avoid starting
expensive jobs without a clear confirmation of dataset, output path, and
key settings.

During training, show available metrics such as: - Training and
validation loss - Tokens processed - Throughput - Elapsed time - Current
step - RAM usage, if measured - Checkpoint state

Use line charts for metrics over time and compact summary values for the
current run. Handle missing validation data explicitly. Never
extrapolate a final loss or estimated completion time without labeling
it as an estimate.

## 10. Benchmarks

The Benchmarks page compares model revisions and baselines under
recorded conditions.

Each benchmark record should include: - Model name and revision -
Dataset and split - Tokenizer - Evaluation method - Seed and relevant
configuration - Hardware and software environment - Parameter count -
Runtime and memory measurements - Quality metrics and uncertainty where
available - Date and artifact references

Support comparison of language modeling, classification, calibration,
selective prediction, tool routing, and efficiency where those
evaluations exist.

Comparisons must be fair: show mismatched tokenizers, training budgets,
or hardware conditions rather than silently treating results as directly
comparable. Allow users to open the full run configuration and raw
result file.

## 11. Memory Explorer

This page is conditional on actual memory functionality in the
model/runtime.

If implemented, it may show: - Current state or memory entries - Entry
type and timestamp - Source and provenance - Read/write events -
Retrieval score or rank - Expiration or retention policy - Whether
memory is ephemeral or persistent

Provide inspect, filter, and clear controls only where supported.
Confirm destructive operations. Do not expose hidden internal state as
meaningful human-readable memory unless the runtime defines how it
should be interpreted.

## 12. Experiments

An experiment is a reproducible record, not merely a named UI session.

Store or display: - Unique run ID - Git commit or source revision -
Model configuration - Dataset identity and split - Seed - Environment
details - Start/end timestamps - Metrics - Logs - Checkpoint and output
paths - Completion, failure, or cancellation status

Allow users to open two runs side by side and compare configuration
differences. Keep original run records immutable; corrections should be
recorded as new versions or annotations.

## 13. Settings

Settings should include:

-   Runtime endpoint and connection test
-   Model and artifact directories
-   Default model
-   Default task mode
-   Appearance: dark, light, or system
-   Reduced motion
-   Log verbosity
-   Diagnostic export
-   Version information

Never transmit prompts, datasets, checkpoints, or telemetry to an
external service by default. Any future remote service integration must
be explicit and opt-in.

## 14. Backend/UI contract

The UI must communicate with a versioned local API rather than importing
Python model code into the frontend.

A conceptual API may include:

-   `GET /api/v1/health` --- runtime status
-   `GET /api/v1/models` --- available models
-   `POST /api/v1/models/load` --- load a model
-   `POST /api/v1/inference` --- execute a supported task
-   `POST /api/v1/jobs` --- create a training/evaluation job, if
    supported
-   `GET /api/v1/jobs/{id}` --- job status and metrics
-   `POST /api/v1/jobs/{id}/cancel` --- request cancellation
-   `GET /api/v1/experiments` --- list recorded runs
-   `GET /api/v1/experiments/{id}` --- run details

These are proposed contracts, not claims about existing endpoints. Align
them with the actual runtime before implementation. Use typed request
and response schemas, explicit error objects, and API versioning. Use
WebSocket or server-sent events for streaming/live metrics if polling
becomes inefficient.

The local server should bind to loopback by default. Validate paths,
requests, and tool permissions. Do not expose an unrestricted local
execution endpoint.

## 15. Suggested project structure

Adapt to the repository rather than forcing a large reorganization
prematurely.

``` text
recamem/
├── src/
│   ├── app/
│   ├── components/
│   │   ├── ui/
│   │   ├── layout/
│   │   ├── status/
│   │   └── charts/
│   ├── features/
│   │   ├── playground/
│   │   ├── architecture/
│   │   ├── training/
│   │   ├── benchmarks/
│   │   ├── memory/
│   │   ├── experiments/
│   │   └── settings/
│   ├── lib/
│   │   ├── api/
│   │   ├── schemas/
│   │   └── utils/
│   ├── styles/
│   └── main.tsx
├── src-tauri/
├── tests/
├── package.json
├── tsconfig.json
├── vite.config.ts
└── design.md
```

This is a target structure for the UI layer. Do not move or rename the
existing Python model package merely to match this diagram.

## 16. Accessibility and interaction requirements

-   All actions must be keyboard accessible.
-   Focus states must be visible.
-   Do not rely on color alone to communicate status.
-   Use semantic labels for inputs and buttons.
-   Provide readable chart summaries or tabular alternatives.
-   Respect reduced-motion settings.
-   Keep errors near the relevant control and provide a detailed
    diagnostic view.
-   Avoid modal dialogs for routine actions.
-   Require confirmation for destructive operations such as deleting
    artifacts or clearing persistent memory.

## 17. Performance requirements

-   Avoid rendering every log line or token as an independent expensive
    component.
-   Virtualize long logs and experiment tables when needed.
-   Throttle live metric updates.
-   Lazy-load large charts and rarely used pages.
-   Do not block the UI thread with model execution.
-   Keep frontend dependencies modest.
-   Make model loading, inference, and training cancellable where the
    backend supports cancellation.

## 18. Security and privacy

-   Bind the local runtime to loopback by default.
-   Do not assume that a local service is safe from all local clients.
-   Validate user-controlled paths and uploaded files.
-   Require explicit permission for tool execution and external network
    access.
-   Keep secrets out of logs and experiment metadata.
-   Make deletion and retention behavior clear.
-   Do not send user data to analytics services by default.

## 19. Implementation phases

### Phase 1 --- UI foundation

-   Establish Tauri 2, React, TypeScript, Vite, Tailwind, and shadcn/ui.
-   Define design tokens, typography, layout, navigation, and status
    components.
-   Add mock data only in a clearly marked development mode.

### Phase 2 --- Runtime connection

-   Implement a typed API client.
-   Connect runtime health and model discovery.
-   Add connection, loading, empty, and error states.
-   Verify that no mock values are shown as live results.

### Phase 3 --- Playground

-   Implement supported inference modes.
-   Add streaming if the backend supports it.
-   Show real timing and decision metadata only when available.

### Phase 4 --- Experiments and benchmarks

-   Read actual run records and artifacts.
-   Add comparison views and reproducibility metadata.
-   Add charts after data schemas are stable.

### Phase 5 --- Advanced research views

-   Architecture inspector
-   Training monitor
-   Memory explorer
-   More detailed profiling

Do not implement all pages before confirming the underlying runtime APIs
and data structures.

## 20. Definition of done

The initial UI is ready when:

-   It launches as a desktop app on the target operating system.
-   The visual system is consistent across pages.
-   It reports runtime connectivity and model state accurately.
-   A user can execute at least one genuinely supported model task.
-   Errors and cancellation are handled cleanly.
-   The UI remains responsive during inference.
-   No fabricated metrics, model capabilities, or decision traces are
    presented as real.
-   The runtime remains independent of the UI.
-   Basic keyboard navigation and contrast requirements are met.
-   Setup and development instructions are documented.

## 21. Explicit non-goals

The initial release is not intended to: - Reproduce ChatGPT's
interface. - Pretend the model has capabilities not implemented in the
backend. - Provide a visual architecture editor that silently changes
model code. - Require a cloud account or hosted inference provider. -
Add a complex plugin system before core workflows work. - Present
unverified performance claims as benchmark results.

## Final direction

Build Recamem as a **local-first neural systems lab**: dark charcoal
surfaces, restrained lime accents, precise typography, compact
information density, and transparent model behavior. Start with a
polished Playground and honest runtime status. Add research dashboards
only as real model, training, and evaluation capabilities become
available.
