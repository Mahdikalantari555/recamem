# system1-decision Specification

## Purpose

Non-autoregressive System-1 decision head that routes user queries to discrete actions (Chat, Recall, Teach, Tool) via hyperspherical cosine argmax in under 3 ms on CPU, with calibrated Shannon entropy gating to reject low-confidence out-of-domain requests.

## Requirements

### Requirement: Zero-shot prototype routing
The system SHALL classify a user query into one of K predefined intent categories using cosine similarity between the query embedding and prototype anchor embeddings, executing in under 3 ms on CPU.

#### Scenario: Intent routing latency
- **WHEN** a query is embedded via the ONNX MiniLM anchor
- **WHEN** cosine similarity is computed against K=5 prototype anchors
- **THEN** the classification SHALL complete in under 3 ms on CPU
- **THEN** the returned intent SHALL be the argmax of the cosine similarity scores

#### Scenario: Zero-shot prototype accuracy
- **WHEN** a query clearly matches one of the K intent categories
- **THEN** the top-1 cosine similarity score SHALL exceed 0.65
- **THEN** the returned intent SHALL match the expected category

### Requirement: Calibrated confidence gating
The system SHALL compute a calibrated confidence score using normalized Shannon entropy and reject ambiguous or out-of-domain requests when confidence falls below a threshold.

#### Scenario: Normalized entropy confidence
- **WHEN** a query produces a softmax distribution over K intent categories
- **THEN** the confidence score SHALL be computed as $1 - \mathcal{H}_{\text{norm}}$ where $\mathcal{H}_{\text{norm}} = \frac{-\sum p_k \ln p_k}{\ln K}$
- **THEN** the confidence score SHALL be bounded between 0.0 and 1.0

#### Scenario: Out-of-domain rejection
- **WHEN** a query has no clear match to any intent category
- **WHEN** the maximum cosine similarity is below 0.35
- **THEN** the system SHALL reject the decision and return an "ambiguous" or "out-of-domain" status
- **THEN** the system SHALL NOT execute any tool or action based on the rejected decision

### Requirement: Closed-form Ridge linear probe for custom tools
The system SHALL train a linear probe mapping query embeddings to custom tool action probabilities using closed-form Ridge regression, completing in under 5 ms on CPU without backpropagation.

#### Scenario: Custom tool training
- **WHEN** user provides N labeled examples of (query, tool_action) pairs
- **THEN** the system SHALL compute the closed-form Ridge solution $W = (X^\top X + \lambda I)^{-1} X^\top Y$
- **THEN** training SHALL complete in under 5 ms on CPU
- **THEN** the resulting probe SHALL be serializable to disk as a safetensors tensor

### Requirement: Structured decision output
The system SHALL return a structured decision object containing the selected action, probability distribution, confidence score, and execution metadata. The memory console SHALL expose a decision panel that displays that object for an operator-supplied query without executing the selected action.

#### Scenario: Structured decision response
- **WHEN** a decision is made
- **THEN** the response SHALL include `action` (string), `probabilities` (list of floats), `confidence` (float), and `latency_ms` (float)
- **THEN** the response SHALL be JSON-serializable

#### Scenario: Console decision panel
- **WHEN** user enters a query in the decision panel
- **THEN** the console SHALL display the selected action, the probability distribution, the confidence, and the measured latency
- **THEN** displaying a decision SHALL NOT execute the action, mutate memory, or call a tool

#### Scenario: Rejected decision is displayed as rejected
- **WHEN** the query is out of domain and the decision head rejects it
- **THEN** the console SHALL show the rejection status and the observed similarity
- **THEN** the panel SHALL NOT show an action as if one had been selected

### Requirement: Option-conditioned scoring over dynamic closed option sets with null option
In addition to static cosine prototype routing (`D1`) and closed-form Ridge probes (`D2`), the system SHALL support scoring arbitrary per-request closed option sets ($K \in [2, 255]$) supplied as natural-language option descriptions (`D3`), encoding the query via attention pooling and each option via the shared sequence core while appending a learned null/out-of-scope option vector $v_0$ to represent "none of the supplied options apply".

#### Scenario: Variable option set size and order invariance
- **WHEN** a query is evaluated against $K$ dynamic textual options in two different permutations
- **THEN** the option-conditioned head SHALL produce identical per-option probabilities (up to permutation and `1e-5` floating-point tolerance) in a single non-autoregressive pass without token decoding
- **WHEN** none of the supplied options match an out-of-scope query
- **THEN** the learned null option $v_0$ SHALL be selectable and map deterministically to abstention / out-of-scope rejection

#### Scenario: Generalization fallback on unseen option label phrasings
- **WHEN** evaluation on held-out unseen option phrasings drops near chance relative to few-shot prototypes
- **THEN** the decision head SHALL support hybrid prototype-anchored option representations (combining option text encoding with few-shot support states) and report which mode ran

### Requirement: Post-hoc temperature calibration and selective risk-coverage metrics
The system SHALL fit post-hoc scalar temperature scaling ($T_{\text{cal}} > 0$) on a held-out calibration split separate from model-selection and test splits, and SHALL evaluate probabilistic calibration separately from accuracy using Negative Log-Likelihood (`NLL`), Brier score, 15-bin equal-mass Adaptive Expected Calibration Error (`adaptive_ece`), and Area Under the Risk-Coverage Curve (`AURC`).

#### Scenario: Temperature scaling preserves argmax while improving calibration
- **WHEN** post-hoc temperature $T_{\text{cal}}$ is fit on `val_cal` and applied to `test_eval` logits
- **THEN** top-1 option argmax predictions SHALL remain unchanged
- **THEN** the report SHALL record pre-scaling and post-scaling `NLL`, `Brier`, `adaptive_ece` (with bin counts), and `AURC` on both in-domain and out-of-distribution splits
