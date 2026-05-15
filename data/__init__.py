# The dataset module pulls in torch; import it lazily so the env (numpy-only)
# can be used even in environments that don't have torch installed.
try:  # pragma: no cover
    from .dataset import (
        FeatureSchema,
        TabularSupplyChainDataset,
        load_dataco_or_synthetic,
    )
except ImportError:  # torch missing -> dataset utilities unavailable
    FeatureSchema = None
    TabularSupplyChainDataset = None
    load_dataco_or_synthetic = None

from .env import MultiEchelonInventoryEnv

__all__ = [
    "FeatureSchema",
    "TabularSupplyChainDataset",
    "load_dataco_or_synthetic",
    "MultiEchelonInventoryEnv",
]
