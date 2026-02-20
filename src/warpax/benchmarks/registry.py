"""Metric registry and ground truth database.

Provides a ``MetricRegistry`` for looking up benchmark spacetimes with
their associated ground truth data.  The registry is a simple Python class
(not an eqx.Module) since it is metadata infrastructure, not traced by JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..geometry.metric import MetricSpecification
from .alcubierre import AlcubierreMetric, GROUND_TRUTH as ALCUBIERRE_GT
from .minkowski import MinkowskiMetric, GROUND_TRUTH as MINKOWSKI_GT
from .schwarzschild import SchwarzschildMetric, GROUND_TRUTH as SCHWARZSCHILD_GT


# ---------------------------------------------------------------------------
# GroundTruth dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """Ground truth data for a benchmark spacetime.

    Parameters
    ----------
    properties : dict[str, Any]
        Arbitrary key-value pairs describing the known analytical properties
        (e.g. kretschner scalar, energy conditions, stress-energy status).
    """

    properties: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.properties

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------


class MetricRegistry:
    """Registry of benchmark spacetimes with ground truth.

    Usage::

        registry = MetricRegistry()
        metric, gt = registry.get("Minkowski")
        print(gt["kretschner"])  # 0.0
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[MetricSpecification, GroundTruth]] = {}

    def register(
        self, metric: MetricSpecification, ground_truth: GroundTruth
    ) -> None:
        """Register a metric with its ground truth data.

        Parameters
        ----------
        metric : MetricSpecification
            The benchmark metric instance.
        ground_truth : GroundTruth
            Associated ground truth properties.
        """
        self._entries[metric.name()] = (metric, ground_truth)

    def get(self, name: str) -> tuple[MetricSpecification, GroundTruth]:
        """Retrieve a metric and its ground truth by name.

        Parameters
        ----------
        name : str
            Human-readable metric name (as returned by ``metric.name()``).

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Metric '{name}' not registered.  "
                f"Available: {self.list_metrics()}"
            )
        return self._entries[name]

    def list_metrics(self) -> list[str]:
        """Return sorted list of registered metric names."""
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# ---------------------------------------------------------------------------
# Default registry with all benchmarks
# ---------------------------------------------------------------------------


def create_default_registry() -> MetricRegistry:
    """Create a registry pre-loaded with all benchmark spacetimes."""
    registry = MetricRegistry()
    registry.register(MinkowskiMetric(), GroundTruth(MINKOWSKI_GT))
    registry.register(SchwarzschildMetric(), GroundTruth(SCHWARZSCHILD_GT))
    registry.register(AlcubierreMetric(), GroundTruth(ALCUBIERRE_GT))
    return registry
