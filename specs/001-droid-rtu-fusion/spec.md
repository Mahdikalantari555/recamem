# Feature Specification: Droid × Raw Byte RTU Fusion

**Feature Branch**: `001-droid-rtu-fusion`

**Created**: 2026-10-11

**Status**: Draft

**Input**: User description: "Implement plans/10_droid_rtu_fusion.md: fuse the droid memory agent with a frozen raw-byte RTU generative backend so answers are both grounded in taught facts and fluent. The RecaMem restructure (plan 09) is deferred to a future repository and is explicitly out of scope."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Answers that are both grounded and fluent (Priority: P1)

An operator teaches the agent a paragraph of domain facts. In the existing modes the
operator must choose: the extractive path echoes the facts verbatim (content without
fluency), and the standalone generative path produces fluent prose that is ungrounded
(content-free). The fused mode removes this choice. A generative component supplies
surface realization, while the agent's taught memory supplies the facts. The question
conditions last, so it dominates what the generative component writes.

**Why this priority**: This is the entire value of the feature. Without it, neither
existing mode answers the operator's real question: "say what I taught you, in your
own words".

**Independent Test**: Teach one paragraph, ask one question on the generative path,
and check the answer contains taught content phrased as connected prose.

**Acceptance Scenarios**:

1. **Given** an agent that has been taught a remote-sensing paragraph, **When** the
   operator asks a question on the fused generative path, **Then** the answer
   contains at least one taught fact and is not a verbatim echo of a single fact.
2. **Given** a fresh agent with no taught facts, **When** the operator asks any
   question on the fused generative path, **Then** the agent returns the existing
   "no information in memory yet" fallback and never generates free text.
3. **Given** a fused agent, **When** the operator asks a question on the extractive
   path, **Then** the answer is byte-identical to the same question on a non-fused
   agent.

---

### User Story 2 - Ranking uses plastic memory, not just surface text (Priority: P2)

The agent keeps a plastic associative memory of what it has seen. Today that memory
never influences which taught facts an answer surfaces. The fused mode adds a bounded
associative bonus to retrieval: it may only re-order facts that already pass the
existing relevance gate, and it must never resurrect an out-of-domain query.

**Why this priority**: It makes the memory system actually contribute to answer
selection, and it is a prerequisite for evidence that the plastic channel is live.

**Independent Test**: Two facts with equal text relevance re-order when the bonus is
active; a cold-start agent's ranking is unchanged.

**Acceptance Scenarios**:

1. **Given** two taught facts with equal relevance to a query, **When** the query is
   processed in fused mode, **Then** the fact more aligned with the plastic memory
   state ranks first.
2. **Given** a freshly created agent with empty plastic memory, **When** any query is
   processed, **Then** the ranking is identical to the ranking without the bonus.
3. **Given** any query with no relevant taught facts, **When** the query is
   processed, **Then** the bonus never promotes an out-of-domain fact above the
   relevance gate into an answer.

---

### User Story 3 - A fused agent is an ordinary agent (Priority: P2)

Operators create, save, and load agent profiles. A fused agent must persist exactly
like any other agent: the generative component's weights and state live inside the
agent's own profile, and the profile records what exists. If the manifest declares a
generative checkpoint and the file is absent, the load must fail loudly instead of
silently running a random component.

**Why this priority**: Silent degradation (a random component producing garbage while
claiming to work) is the failure mode this feature's honesty claims exist to prevent.

**Independent Test**: Save a fused agent, load it, and confirm the loaded component
behaves identically to the saved one; delete the declared checkpoint and confirm load
raises a clear error.

**Acceptance Scenarios**:

1. **Given** a saved fused agent, **When** the profile is loaded into a fresh
   session, **Then** the generative component's weights and state buffers equal the
   saved values and answers behave the same way.
2. **Given** a profile whose manifest declares a generative checkpoint, **When** the
   checkpoint file is missing on disk, **Then** loading the profile raises a clear
   error and does not fall back to a random-initialized component.
3. **Given** an operator who supplies both the fused backend and the existing large
   language model backend at construction, **When** the agent is constructed, **Then**
   the system raises an error instead of silently choosing one.
4. **Given** an operator creating an agent, **When** they pass a single
   "use fused backend" flag, **Then** the agent is created with the fused backend
   wired and ready.

---

### User Story 4 - Operators can see which path answered (Priority: P3)

The operator console shows engine modes. A third mode, "fused", appears beside the
existing two, and the memory console reports, per answer, whether it was generative
or extractive, plus the generative component's configuration and checkpoint status.
The standalone generative mode keeps its own controls and behavior unchanged.

**Why this priority**: It makes the new behavior observable and debuggable without
reading logs, and it protects the existing standalone mode from drift.

**Independent Test**: Switch to fused mode in the console, ask both a hit and a miss,
and confirm the badge reports the path used and the component's configuration.

**Acceptance Scenarios**:

1. **Given** the console, **When** the operator selects the fused mode, **Then** a
   third engine option exists beside the existing two and the extraction path
   behaves exactly as in the plain droid mode.
2. **Given** a fused agent, **When** an answer is produced, **Then** the console
   reports whether it was generative or extractive and shows the generative
   component's configuration and checkpoint status.
3. **Given** the standalone generative mode, **When** the operator uses it after
   this change, **Then** its behavior and controls are identical to before.

---

### Edge Cases

- What happens when the fused generative path receives an empty or whitespace-only
  question? The system returns an empty answer without contacting the generative
  component.
- How does the system handle non-ASCII questions and answers? Decoding failures must
  never crash the path; output is always a valid string.
- What happens when the answer stream is truncated mid-character? The partial output
  is decoded leniently and returned as valid text.
- What happens when no generative checkpoint was ever trained and the operator runs
  fused mode directly (not via a saved profile)? The system still runs, with a
  random-initialized component, and the console surfaces the untrained status so the
  operator is not misled about quality.
- What happens when a fused answer would take too long? Prompt and output are capped
  so a single answer stays within the measured latency budget recorded in the plan.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST offer a third engine mode ("fused") beside the
  existing two, in which the generative component is conditioned on retrieved taught
  facts and the user's question.
- **FR-002**: In fused mode, the system MUST place the user's question last in the
  prepared conditioning context, after the retrieved facts, so the question dominates
  the generative state.
- **FR-003**: In fused mode, the generative component's weights and internal state
  MUST remain frozen at all times: all learning happens only in the agent's plastic
  memory. Any future variant that trains the generative component must run on a
  guarded copy with snapshot and rollback, never on the inference component.
- **FR-004**: In fused mode, when retrieval returns no relevant facts, the system
  MUST return the existing "no information in memory yet" fallback and MUST NOT
  generate free text. Grounded silence beats fluent hallucination.
- **FR-005**: The extractive (fast) path in fused mode MUST produce byte-identical
  answers to the same path on a non-fused agent.
- **FR-006**: The system MUST persist the fused generative component's weights and
  state inside the agent's own profile directory, alongside the existing profile
  files, and record the checkpoint in the profile manifest.
- **FR-007**: On profile load, when the manifest declares a generative checkpoint
  that is absent on disk, the system MUST raise a clear error and MUST NOT
  substitute a random-initialized component.
- **FR-008**: When both the fused backend and the existing large-language-model
  backend are supplied at agent construction, the system MUST raise an error instead
  of silently choosing one.
- **FR-009**: Operators MUST be able to create a fused agent with a single
  construction flag that wires the fused backend.
- **FR-010**: Retrieval MUST incorporate a bounded associative bonus derived from
  plastic memory that only re-orders candidates that already pass the existing
  relevance gate. The gate stays authoritative.
- **FR-011**: With empty plastic memory (cold start), the associative bonus MUST NOT
  change any ranking, including answers produced through the generative path.
- **FR-012**: The operator console MUST report, for every fused answer, whether it
  was produced by the generative or the extractive path, and MUST display the
  generative component's configuration and checkpoint status.
- **FR-013**: The standalone generative mode's controls and behavior MUST remain
  byte-identical to their pre-change state.
- **FR-014**: Generation MUST be bounded by a prompt-length cap and an output-length
  cap, so a single fused answer completes within the measured latency budget
  recorded in the plan file.
- **FR-015**: The generative path MUST accept any Unicode input and MUST always
  return a valid string, including when the output is truncated mid-character.

### Key Entities *(include if feature involves data)*

- **Fused Agent Profile**: one agent's saved state: the profile manifest (what
  components exist), the knowledge store, the plastic memory, and the generative
  component checkpoint. A fused profile adds exactly one new file (the generative
  checkpoint) and three manifest fields (checkpoint reference, conditioning strategy
  pin, component dimensions) to the existing layout.
- **Answer Path**: the route an answer took — extractive (verbatim taught facts,
  unchanged) or generative (fluent prose conditioned on taught facts, new). Every
  fused answer carries this label for the console badge.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After teaching one paragraph, 100% of fused generative answers (across
  repeated runs) contain at least one taught fact.
- **SC-002**: A fresh agent with no taught facts produces the standard "no
  information" fallback in 100% of fused generative attempts; generated free text
  appears in 0% of those attempts.
- **SC-003**: Extractive-path answers are byte-identical between fused and non-fused
  agents across the full existing test suite.
- **SC-004**: The generative component's weights and state buffers are unchanged
  (identical checksums) after 10 or more consecutive fused answers.
- **SC-005**: A fused agent's profile round-trips: after save and load, the
  generative component's weights and state equal the saved values, and answers
  behave the same.
- **SC-006**: The complete pre-existing test suite passes with zero modifications to
  existing contracts.
- **SC-007**: The documented remote-sensing end-to-end verification passes with the
  fused backend attached.
- **SC-008**: A single fused answer completes within 10 seconds on the project's
  reference machine; the measured ceiling is recorded in the plan file before the
  feature is called complete.
- **SC-009**: A cold-start agent's ranking equals the pre-change ranking in 100% of
  sampled queries (no cold-start regression from the associative bonus).
- **SC-010**: The associative bonus never promotes a below-gate candidate into a
  visible answer, across all sampled queries.

## Assumptions

- Users are the existing operators of this local agent system: one person operates
  one agent from the same machine. The fused generative path is opt-in (per-question
  slow command or per-agent mode choice), not a default.
- The generative component is a pre-existing, separately trained byte-level
  micro-model component. This feature does not include any training workflow for it.
  Until a trained checkpoint exists, fused mode runs with a random-initialized
  component, and the console surfaces that status (answer quality is then low and
  documented as such).
- Sampling is stochastic, so end-to-end content checks assert on substring presence
  in taught facts; equality checks for the frozen behavior assert on internal state
  tensors and checksums, never on generated text.
- The generative component's dimensions match the agent's embedding dimension so the
  associative bonus and any state bridging are dimensionally well-defined.
- The RecaMem restructure (plan 09: package rename, god-class decomposition, new
  forgetting/consolidation/benchmark subsystems) is deferred to a separate future
  repository. This feature introduces no renames and touches only the new backend
  module, the engine, the profile manager, the console, and documentation.
- The existing remote-sensing end-to-end fixture is the regression anchor for
  retrieval quality; it must stay green through this change.
