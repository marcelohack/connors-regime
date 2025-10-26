"""
Base service class providing common patterns and utilities
"""

import logging
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict


class BaseService(ABC):
    """Base class for all services providing common functionality"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_required_params(
        self, params: Dict[str, Any], required: list[str]
    ) -> None:
        """Validate that all required parameters are present and not None or empty"""
        missing = [
            param
            for param in required
            if param not in params or params[param] is None or params[param] == ""
        ]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    def _safe_execute(
        self,
        func: Callable[..., Any],
        error_msg: str = "Operation failed",
        **kwargs: Any,
    ) -> Any:
        """Safely execute a function with error handling"""
        try:
            return func(**kwargs)
        except Exception as e:
            self.logger.error(f"{error_msg}: {e}")
            raise

    def _ensure_directory_exists(self, directory_path: Path) -> None:
        """Create directory if it doesn't exist"""
        directory_path.mkdir(parents=True, exist_ok=True)
