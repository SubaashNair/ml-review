"""Shared result-data helpers."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def result_data(result: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """Return the mapping payload from a raw dict or MLReview result object."""
    if isinstance(result, Mapping):
        return result
    raise TypeError("Expected an MLReview result object or result mapping.")


def result_dict(result: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Copy an MLReview result payload into a plain dictionary."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return deepcopy(dict(result_data(result)))
