"""Compatibility wrapper that forwards to the unified main module."""
from __future__ import annotations

from main import ModelManager, StemModel  # re-export

__all__ = ["ModelManager", "StemModel"]
