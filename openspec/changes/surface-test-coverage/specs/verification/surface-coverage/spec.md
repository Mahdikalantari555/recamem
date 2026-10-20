# Spec Delta

## Purpose

Define the verification contract for module-level surface coverage: every shipped module has at least one test file that exercises its public contract, and tests assert the return contracts the implementation actually provides.

## ADDED Requirements

### Requirement: Untested module coverage
Every Python module under `src/` other than `__init__.py` SHALL be mapped to a test file that imports it and exercises at least one public behavior. The mapping SHALL be explicit in the test suite, and a module absent from the mapping SHALL fail the coverage test with the module path and its expected test filename.

#### Scenario: New module without a test
- **WHEN** a module is added under `src/` and no test mapping entry exists for it
- **THEN** the coverage test SHALL fail
- **THEN** the failure message SHALL name the module and the test file that is expected to cover it

#### Scenario: Mapped test file is missing or empty
- **WHEN** the mapping names a test file that does not exist or contains no test function
- **THEN** the coverage test SHALL fail

#### Scenario: Full coverage is the steady state
- **WHEN** the change is complete
- **THEN** the untested-module list SHALL be empty
- **THEN** the mapping SHALL include the modules covered by pre-existing test files (`droid.py`, `rtu.py`, `droid_manager.py`, `knowledge_store.py`, `onnx_anchor.py`, `teacher.py`, `data.py`, `byte_rtu_backend.py`)

### Requirement: Test contract conformance
A test SHALL assert the contract the implementation provides, not a convenient destructuring of it. Where a function returns a tuple, the test SHALL unpack every element with its documented meaning, and SHALL read secondary outputs (such as a loss recorded on the object) from the attribute that holds them.

#### Scenario: Forward return contract
- **WHEN** a test calls `Model.forward` and inspects the result
- **THEN** it SHALL treat the return value as `(sampled_byte, stop_value)`
- **THEN** it SHALL read the training loss from `model.last_loss`
- **THEN** it SHALL NOT compare a loss attribute against the stop value

#### Scenario: Loss value is pinned
- **WHEN** the cross-entropy-only path is tested
- **THEN** the expected cross-entropy SHALL be computed independently and compared to `model.last_loss` with a tolerance
- **THEN** `step_count` and the trace-buffer advance SHALL still be asserted

#### Scenario: Signature is not changed for test convenience
- **WHEN** a test needs the loss
- **THEN** the implementation SHALL NOT change `Model.forward`'s return signature
- **THEN** the MLX reference's `(sample, stop)` return SHALL remain the contract
