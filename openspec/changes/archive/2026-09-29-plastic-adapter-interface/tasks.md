# Tasks

## 1. PlasticAdapter Interface & Registry

- [x] 1.1 Create `src/model/plastic_adapter.py` with `PlasticAdapter` ABC (inherits `nn.Module`) and 6 abstract methods: `update`, `retrieve`, `predict_next`, `compute_surprise`, `state_dict`, `load_state_dict`
- [x] 1.2 Add optional methods with default implementations: `consolidate()`, `get_health()`, `create_snapshot()`, `restore_snapshot()`
- [x] 1.3 Implement registry: `_ADAPTER_REGISTRY`, `register_adapter(name, cls)`, `get_adapter(name, **kwargs)`, `list_adapters()`
- [x] 1.4 Implement entry point auto-discovery: scan `importlib.metadata.entry_points(group="tmt.plastic_adapters")` on module import
- [x] 1.5 Register built-in: `register_adapter("associative_rtu", PlasticAssociativeRTU)` in `rtu.py`
- [x] 1.6 Verify: `tests/test_plastic_adapter.py` tests ABC compliance, registry, entry points

## 2. PlasticAssociativeRTU Implements Interface

- [x] 2.1 Update `PlasticAssociativeRTU` class definition: `class PlasticAssociativeRTU(PlasticAdapter)`
- [x] 2.2 Verify all 6 abstract methods implemented (map existing: `update_associative_memory`→`update`, `retrieve`→`retrieve`, `predict_next`, `compute_surprise`, `state_dict`, `load_state_dict`)
- [x] 2.3 Add `state_dict()` returning `{**super().state_dict(), "_adapter_version": 1, "_adapter_type": "associative_rtu"}`
- [x] 2.4 Add `load_state_dict()` handling version migration
- [x] 2.5 Verify: `tests/test_rtu.py` passes with interface checks

## 3. DroidEngine Integration

- [x] 3.1 Update `DroidEngine.__init__` to accept `plastic_adapter` instance or create via `get_adapter(config["plastic_adapter"], **config.get("adapter_kwargs", {}))`
- [x] 3.2 Replace direct `self.memory.method()` calls with `self.memory.method()` (interface same, but type is now `PlasticAdapter`)
- [x] 3.3 Update type hints: `self.memory: PlasticAdapter`
- [x] 3.4 Verify: `tests/test_droid_engine.py` works with interface

## 4. Profile & Manager Integration

- [x] 4.1 Update `DroidManager.create_droid()` to accept `plastic_adapter="associative_rtu"`, `adapter_kwargs={}`
- [x] 4.2 Update `config.json` schema: add `plastic_adapter`, `adapter_kwargs` fields
- [x] 4.3 Update `save_profile()` to write `plastic_adapter` and `adapter_kwargs`
- [x] 4.4 Update `load_profile()` to read and instantiate adapter via registry
- [x] 4.5 Verify: manager test creates droid with explicit adapter config

## 5. Tests & Validation

- [x] 5.1 Create `tests/test_plastic_adapter.py`: ABC compliance, registry CRUD, entry point mock, state_dict round-trip
- [x] 5.2 Extend `tests/test_rtu.py`: verify `PlasticAssociativeRTU` passes ABC checks
- [x] 5.3 Extend `tests/test_droid_engine.py`: test with mock adapter implementing interface
- [x] 5.4 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [x] 5.5 Benchmark: adapter instantiation < 10ms, no performance regression vs direct class