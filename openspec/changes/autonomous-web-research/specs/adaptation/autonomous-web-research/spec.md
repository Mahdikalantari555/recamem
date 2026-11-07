## ADDED Requirements

### Requirement: Opt-in bounded research sessions
The system SHALL offer explicitly enabled, read-only research sessions with bounded search and navigation, controlled by deterministic policies. The system SHALL state whether it uses an autonomous planner or a deterministic fallback and SHALL NOT assume that a smoke-trained model can plan.

#### Scenario: No planner configured
- **WHEN** a capable planner is unavailable
- **THEN** the system SHALL report the limitation and either offer explicitly labelled bounded static research or stop without fabricating autonomous actions.

#### Scenario: Cancelled session
- **WHEN** a user cancels an active session
- **THEN** the system SHALL stop scheduling requests, terminate active work where supported and prevent unapproved memory writes.

### Requirement: Enforced public-only network boundary
The system SHALL enforce a public-only destination policy for all network requests including search, robots, redirects, browser subresources, frames and auxiliary fetches, pin validated destinations against DNS rebinding and preserve TLS hostname verification. Browser mode SHALL fail closed if its entire network traffic cannot be isolated by the boundary.

#### Scenario: Redirect or browser subresource reaches a private destination
- **WHEN** a public page attempts to access a private or loopback address via a redirect, subresource or DNS rebinding
- **THEN** the request SHALL be blocked and reported without revealing internal content.

### Requirement: Source evidence and gated memory
The system SHALL treat fetched content as untrusted data, return source URLs and extraction provenance for claims, and automatically save eligible sourced candidate facts only when the user has explicitly enabled automatic saving; a review-before-saving mode SHALL also be available. Existing profiles SHALL NOT be silently migrated.

#### Scenario: Malicious source requests a tool action
- **WHEN** fetched content asks the agent to disclose secrets, visit internal services or change its instructions
- **THEN** it SHALL NOT grant permissions or execute the requested action, and no page instruction SHALL authorize a memory write; only the user-configured save policy may do so.

### Requirement: Legal and resource controls
The system SHALL enforce configured robots policies, per-host pacing, request and time limits, and user-visible structured failures; it SHALL NOT bypass login, CAPTCHA, paywalls or access controls.

#### Scenario: Robots policy unavailable
- **WHEN** the applicable robots policy cannot be determined
- **THEN** the session SHALL stop or skip that host and explain the reason instead of proceeding silently.

### Requirement: Honest research interface
The companion SHALL expose session status, sources, candidate facts, save policy, optional approvals and errors without presenting unverified claims as established knowledge or missing capabilities as active.

#### Scenario: Optional browser unavailable
- **WHEN** dynamic rendering is requested but the optional browser or required isolation is unavailable
- **THEN** the interface SHALL report the missing capability and SHALL NOT pretend rendering succeeded.
