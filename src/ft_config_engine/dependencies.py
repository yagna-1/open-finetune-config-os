from __future__ import annotations

from .constants import DEPENDENCY_MATRIX


def dependency_stack_for_platform(platform: str) -> list[str]:
    key = platform.strip().lower()
    if key not in DEPENDENCY_MATRIX:
        raise ValueError(f"unsupported platform for dependency stack: {platform}")
    return DEPENDENCY_MATRIX[key][:]
