# Tasks

- [ ] 1.1 Map paper equations and initialization to tensor shapes and document deviations and reference attribution.
- [x] 1.2 Implement CPU sequential recurrence, gates, readout variants and convolution cache.
- [x] 1.3 Integrate opt-in core registry, actual parameter accounting and configuration without changing C4 defaults.
- [x] 1.4 Test sequence versus step and chunk continuation with nonzero state, multiple batches, convolution cache and reset.
- [ ] 1.5 Test causal behavior, finite outputs and gradients against a float64 reference across challenging decays.
- [ ] 1.6 Benchmark parameter-matched C4 and Muir with shared CPU data, budgets, seeds and readout ablations; record performance and hardware.
- [ ] 1.7 If warranted, implement a parallel CPU path and test state, outputs and gradients against the sequential oracle, including partial chunks.
- [ ] 1.8 Document checkpoints, limitations and experimental status; run relevant core and OpenSpec tests.
