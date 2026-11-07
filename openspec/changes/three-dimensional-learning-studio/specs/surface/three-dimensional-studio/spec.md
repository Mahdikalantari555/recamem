## ADDED Requirements

### Requirement: Accessible optional 3D workspace
The companion SHALL provide an optional 3D visual layer without requiring WebGL or animation to access equivalent learning, evidence and control functions.

#### Scenario: WebGL unavailable
- **WHEN** WebGL initialization fails or is disabled
- **THEN** a usable semantic 2D interface and equivalent controls remain available without falsely claiming that a 3D scene is active.

#### Scenario: Reduced motion requested
- **WHEN** the user or operating system requests reduced motion
- **THEN** automatic decorative motion stops and the same information and controls remain available.

### Requirement: Measured and attributable visualization
The interface SHALL distinguish decorative imagery from backend-derived data and show the actual status, provenance and freshness of data-driven representations.

#### Scenario: Background learning is blocked
- **WHEN** the backend reports background learning blocked or unavailable
- **THEN** the interface displays the reported status and SHALL NOT depict decorative motion as evidence of active training.

#### Scenario: Evidence is selected
- **WHEN** an authorized evidence record is shown
- **THEN** its available provenance and uncertainty are available in a semantic HTML view without implying that retrieval verified the generated answer or changed model weights.

### Requirement: Resource-aware rendering lifecycle
The interface SHALL bound and release 3D resources, suspend unneeded rendering and preserve the 2D experience if rendering fails or context is lost.

#### Scenario: Context loss
- **WHEN** the graphics context is lost
- **THEN** the interface stops the affected animation, preserves core controls and either recovers safely or offers the 2D fallback without reporting a false recovery.

### Requirement: Local-first scene and data safety
The scene SHALL use only authorized existing backend data and SHALL NOT initiate model training, research, network access to third parties or memory writes merely by being opened or interacted with.

#### Scenario: Opening the observatory
- **WHEN** the user opens the 3D view
- **THEN** it does not enable autonomous research or training, and the existing permissions remain unchanged.
