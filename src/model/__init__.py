
try:
    from recamem.plastic_adapter import PlasticAdapter, register_adapter, get_adapter, list_adapters
except Exception:  # noqa: BLE001
    try:
        from .plastic_adapter import PlasticAdapter, register_adapter, get_adapter, list_adapters
    except Exception:  # noqa: BLE001
        PlasticAdapter = None
        register_adapter = None
        get_adapter = None
        list_adapters = lambda: []

from src.model.rtu import (
    Model,
    Layer,
    Encoder,
    Decoder,
    CustomAdamW,
    loss_variance,
    loss_pred_mse,
    loss_crossentropy,
    loss_stop_mse,
    compute_losses,
    RTUMemoryBlock,
    PlasticAssociativeRTU,
)

try:
    from recamem.engine import DroidEngine
except Exception:  # noqa: BLE001
    try:
        from .droid import DroidEngine
    except Exception:  # noqa: BLE001
        DroidEngine = None

try:
    from recamem.manager import DroidManager
except Exception:  # noqa: BLE001
    try:
        from .droid_manager import DroidManager
    except Exception:  # noqa: BLE001
        DroidManager = None

try:
    from recamem.episodic import KnowledgeStore
except Exception:  # noqa: BLE001
    try:
        from .knowledge_store import KnowledgeStore
    except Exception:  # noqa: BLE001
        KnowledgeStore = None

try:
    from recamem.decision_head import DecisionHead
except Exception:  # noqa: BLE001
    try:
        from .decision_head import DecisionHead
    except Exception:  # noqa: BLE001
        DecisionHead = None

try:
    from recamem.package import DroidPackage
except Exception:  # noqa: BLE001
    try:
        from .droid_package import DroidPackage
    except Exception:  # noqa: BLE001
        DroidPackage = None

try:
    from recamem.gguf import GgufBackend
except Exception:  # noqa: BLE001
    try:
        from .gguf_backend import GgufBackend
    except Exception:  # noqa: BLE001
        GgufBackend = None

try:
    from recamem.lifecycle import (
        SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver,
        RetrievalPolicy, ContinuousStream
    )
except Exception:  # noqa: BLE001
    try:
        from .memory_lifecycle import (
            SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver,
            RetrievalPolicy, ContinuousStream
        )
    except Exception:  # noqa: BLE001
        SnapshotManager = None
        MergeEngine = None
        SemanticStore = None
        ContradictionResolver = None
        RetrievalPolicy = None
        ContinuousStream = None

try:
    from recamem.injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
except Exception:  # noqa: BLE001
    try:
        from .memory_injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
    except Exception:  # noqa: BLE001
        inject_prompt = None
        inject_hidden = None
        inject_kv_cache = None
        select_strategy = None

try:
    from recamem.confidence import (
        retrieval_confidence, generation_confidence, fuse_confidence,
        confidence_tier, estimate_confidence
    )
except Exception:  # noqa: BLE001
    try:
        from .confidence import (
            retrieval_confidence, generation_confidence, fuse_confidence,
            confidence_tier, estimate_confidence
        )
    except Exception:  # noqa: BLE001
        retrieval_confidence = None
        generation_confidence = None
        fuse_confidence = None
        confidence_tier = None
        estimate_confidence = None


try:
    from recamem.segmenter import extract_propositions, extract_propositions_with_fallback
except Exception:  # noqa: BLE001
    try:
        from .propositions import extract_propositions, extract_propositions_with_fallback
    except Exception:  # noqa: BLE001
        extract_propositions = None
        extract_propositions_with_fallback = None

__all__ = [
    "Model",
    "Layer",
    "Encoder",
    "Decoder",
    "CustomAdamW",
    "loss_variance",
    "loss_pred_mse",
    "loss_crossentropy",
    "loss_stop_mse",
    "compute_losses",
    "RTUMemoryBlock",
    "PlasticAssociativeRTU",
    "PlasticAdapter",
    "DroidEngine",
    "DroidManager",
    "KnowledgeStore",
    "DecisionHead",
    "DroidPackage",
    "GgufBackend",
    "SnapshotManager",
    "MergeEngine",
    "SemanticStore",
    "ContradictionResolver",
    "RetrievalPolicy",
    "ContinuousStream",
    "extract_propositions",
    "extract_propositions_with_fallback",
]
