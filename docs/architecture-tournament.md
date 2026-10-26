# RecaMem Core Architecture Tournament (`recamem-core-architecture-tournament`)

`scripts/run_core_tournament.py` executes the controlled 4-stage elimination, interference, and Leave-One-Out (`LOO`) pruning tournament across all 24+ candidate arms from the research report across five orthogonal design tracks (`C0`–`C6`, `O1`–`O4` + `Tok`, `D1`–`D4`, `R1`–`R4`, `M0`–`M5`).

## Stage Gates

1. **Stage 0 (Milestone M0)**: Verifies `±2.0%` parameter parity across all seven cores (`C0`–`C6`), lossless UTF-8 round-trips (`V=256` bytes and `V=4096` compact BPE), and zero `train`/`val_cal`/`test_eval` split contamination.
2. **Stage 1 (Mechanistic Falsification)**:
   - Cores must achieve `MQAR-16 >= 0.75` without NaN divergence.
   - Auxiliary objectives (`O2`, `O3`) must beat pure `O1` (`ce_eos`) by `ΔBPB >= 0.015` (or `+2.0` pts linear-probe Macro-F1 without BPB loss).
   - `O4` (`online_rtrl`) must match `chunked_bptt` BPB within `0.02` and improve throughput while supporting the winning core class.
3. **Stage 2 (Multi-Seed Per-Track Ranking)**: Evaluates surviving finalists across multiple seeds with 3-point LR micro-sweeps on `val_cal`, Pareto frontiers, and 10,000-resample paired bootstrap 95% confidence intervals.
4. **Stage 3 (Shared-vs-Split Interference Gate)**: Compares unified multi-task training (`interleaved 0.80/0.15/0.05` vs `summed` loss) against standalone models and Equal-Budget `D4` (`SplitSpecialistSystem`, `±2.0%` total parameter parity). Selects `unified` when `ΔBPB <= +3.0%` and `ΔMacro-F1 >= -2.0` pts; automatically falls back to `split_specialist` if `ΔBPB > +5.0%` or `ΔMacro-F1 < -5.0` pts.
5. **Stage 4 (Leave-One-Out Pruning)**: Removes any optional complexity component whose marginal utility contribution is `< 0.005` (`0.5%`).

## CLI Usage

```bash
# Inspect CLI flags
PYTHONPATH=. python scripts/run_core_tournament.py --help

# Dry-run parameter matching & split verification
PYTHONPATH=. python scripts/run_core_tournament.py --dry-run --tier micro

# Full tournament execution writing canonical RecaMem core artifacts
PYTHONPATH=. python scripts/run_core_tournament.py --tier micro --seeds 1337,2026,4242
```

## Canonical RecaMem Core Integration

Downstream `recamem` code (`plans/09_recamem_plan.md`) loads the verified winning architecture through a single factory call without hardcoding core IDs:

```python
from src.model.recamem_core import build_recamem_core

bundle = build_recamem_core("configs/recamem_selected_core.json")
bundle.teach("Sentinel-2 revisit period is 5 days", metadata={"entity_key": "sentinel2 revisit"})
recalled = bundle.recall("sentinel2 revisit")
decision = bundle.decide("Adjust perigee altitude", ["Orbital maneuver burn", "Mask clouds"])
routing = bundle.route("Compute 144 * 29 with calculator")
reply = bundle.generate("Summarize orbital radiometry:", max_new_tokens=24)
```
