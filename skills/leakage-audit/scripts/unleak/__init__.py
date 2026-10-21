"""unleak: find data leakage in machine-learning code."""

from .checker import Finding, check_file, check_text

__version__ = "0.1.0"
__all__ = ["Finding", "check_file", "check_text", "__version__"]
