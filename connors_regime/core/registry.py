"""
Component Registry for Regime Detection Methods

Minimal registry focused on regime detection method registration.
Enables external custom detector registration via decorator pattern.
"""

from typing import Any, Callable, Dict, List, Type


class RegimeMethodRegistry:
    """Registry for regime detection methods"""

    def __init__(self) -> None:
        self._regime_methods: Dict[str, Any] = {}  # External regime detection methods

    def register_regime_method(self, name: str) -> Callable[[Type], Type]:
        """Register an external market regime detection method"""

        def decorator(cls: Type) -> Type:
            self._regime_methods[name] = cls
            cls._registry_name = name
            return cls

        return decorator

    def get_regime_method(self, name: str) -> Any:
        """Get a regime detection method by name"""
        if name not in self._regime_methods:
            raise ValueError(
                f"Regime method '{name}' not found. Available: {list(self._regime_methods.keys())}"
            )
        return self._regime_methods[name]

    def list_regime_methods(self) -> List[str]:
        """List available external regime detection methods"""
        return list(self._regime_methods.keys())


# Global registry instance
registry = RegimeMethodRegistry()
