"""Regime method registry — delegates to the unified ComponentRegistry in connors-core.

All regime method registrations are stored in connors-core's central storage backend.
"""

from connors_core.core.registry import ComponentRegistry, registry

# Backward-compatible alias: ``RegimeMethodRegistry()`` creates a ComponentRegistry
# that has register_regime_method, get_regime_method, list_regime_methods, etc.
RegimeMethodRegistry = ComponentRegistry

__all__ = ["RegimeMethodRegistry", "registry"]
