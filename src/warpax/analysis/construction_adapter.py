"""Uniform adapter for certifying heterogeneous positive-energy warp constructions.

The warp-drive metrics in :mod:`warpax.metrics` do not share a constructor: the
compact family takes ``cls(v_s=, R=1, sigma=8)`` while the source-prescribed
shells take ``R_1, R_2, ...`` through factory functions and (for the T-shell) a
matter tilt ``v_0`` rather than a shift speed ``v_s``. This module wraps each
construction behind a single :class:`ConstructionSpec` so one certification pipeline can flow
all of them through the frame-independent certifier
(:func:`..energy_conditions.frame_free.certify_grid_frame_free`) and the
all-observer verification (:mod:`.invariant_verification`) at matched, wall-
resolved settings.

A resolution gate (:func:`wall_cells`) operationalises the paper's
"never report an unresolved wall" rule: a construction whose wall spans fewer than
``MIN_WALL_CELLS`` grid cells is flagged ``resolved=False`` and its certification
numbers are withheld.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..benchmarks import AlcubierreMetric
from ..geometry.metric import MetricSpecification
from ..metrics import (
    GarattiniMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellPhysical,
    fuchs_default,
    sshell_default,
    tshell_default,
)

MIN_WALL_CELLS = 4.0


@dataclass(frozen=True)
class ConstructionSpec:
    """A positive-energy warp construction wrapped for uniform certification."""

    name: str
    build: Callable[[float], MetricSpecification]
    default_speed: float
    speed_param: str  # "v_s" or "v_0"
    bounds: tuple[tuple[float, float], ...]
    grid_n: int
    is_comoving: bool
    # Clustering strength of the graded grid this construction is certified on.
    # The resolution witness must be measured on that same grid, so it lives
    # here rather than at the call site.
    cluster_a: float = 2.0
    # Published energy-condition claim (community summary), for the
    # certified-vs-claimed agreement column. Free-text, not a refutation.
    claim: str = ""
    # Every physical parameter that defines this construction, recorded so the
    # panel is reproducible from the output alone rather than from the source.
    params: dict = field(default_factory=dict)
    # Characteristic wall radius R_c. Curvature carries 1/L^2, so cross-
    # construction stress margins are only comparable after multiplying by
    # R_c^2; and the radial grid is clustered on it.
    wall_radius: float = 1.0
    # Outer radius of the sampled ball for the axisymmetric ladder.
    r_max: float = 3.0
    # Axial position of the bubble at t = 0. The axisymmetric reduction is exact
    # about either this point or the origin, but the radial clustering only lands
    # on the wall when the two agree. A callable is resolved against the built
    # metric, which the Garattini-Zatrimaylov drive needs: its bubble sits at
    # r_0 = v_s / H, so the centre moves with the parameters.
    grid_center: float | Callable[[MetricSpecification], float] = 0.0

    def center_of(self, metric: MetricSpecification) -> float:
        gc = self.grid_center
        return float(gc(metric)) if callable(gc) else float(gc)

    extra: dict = field(default_factory=dict)

    def metric(self, speed: float | None = None) -> MetricSpecification:
        return self.build(self.default_speed if speed is None else speed)


def _alcubierre(v_s):
    return AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0)


def _rodal(v_s):
    return RodalMetric(v_s=v_s, R=1.0, sigma=8.0)


def _natario(v_s):
    return NatarioMetric(v_s=v_s, R=1.0, sigma=8.0)


def _vdb(v_s):
    return VanDenBroeckMetric(v_s=v_s, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0)


def _garattini(v_s):
    # de Sitter background warp bubble; H is matched so that the speed
    # v_s = H * R sits at the Garattini-Zatrimaylov averaged-condition regime.
    return GarattiniMetric(v_s=v_s, R=1.0, sigma=8.0, H=v_s)


def construction_registry() -> dict[str, ConstructionSpec]:
    """All registered constructions keyed by name (compact references + shells)."""
    specs = [
        # Compact references (baseline + the irrotational global-Type-I claim).
        # grid_n = 64: the worst-case resolution witness gives 4.72 cells across
        # the 10-90% wall, against 3.57 at N = 50, below MIN_WALL_CELLS.
        ConstructionSpec(
            "Alcubierre",
            _alcubierre,
            0.5,
            "v_s",
            ((-3.0, 3.0),) * 3,
            64,
            is_comoving=True,
            claim="baseline; NEC/WEC violated for all observers",
            params={"v_s": 0.5, "R": 1.0, "sigma": 8.0},
            wall_radius=1.0,
            r_max=3.0,
        ),
        ConstructionSpec(
            "Rodal",
            _rodal,
            0.5,
            "v_s",
            ((-3.0, 3.0),) * 3,
            64,
            is_comoving=True,
            claim="global Hawking-Ellis Type I; reduced (not eliminated) violations",
            params={"v_s": 0.5, "R": 1.0, "sigma": 8.0},
            wall_radius=1.0,
            r_max=3.0,
        ),
        # Additional positive-energy / source-prescribed constructions.
        ConstructionSpec(
            "Fuchs",
            lambda v: fuchs_default(v_s=v),
            0.02,
            "v_s",
            ((-25.0, 25.0),) * 3,
            60,
            is_comoving=True,
            claim="constant-velocity shell satisfying all energy conditions (arXiv:2405.02709)",
            params={
                "v_s": 0.02,
                "R_1": 10.0,
                "R_2": 20.0,
                "R_b": 1.0,
                "r_s_param": 6.668692,
                "kernel": "moving_average",
            },
            wall_radius=15.0,
            r_max=25.0,
        ),
        ConstructionSpec(
            "WarpShell",
            lambda v: WarpShellPhysical(v_s=v),
            0.02,
            "v_s",
            ((-25.0, 25.0),) * 3,
            60,
            is_comoving=True,
            wall_radius=15.0,
            r_max=25.0,
            claim="Bobrick-Martire / Fell-Heisenberg shell; WEC/NEC/SEC at wall, DEC violated",
        ),
        ConstructionSpec(
            # N = 192, not 64: the bubble co-moves with the Hubble flow and sits
            # at r_0 = v_s/H = 1, so origin-anchored clustering spends its
            # resolution on the centre. At N = 64 the wall spans 1.42 cells; the
            # four-cell floor needs N >= 181.
            "Garattini",
            _garattini,
            0.1,
            "v_s",
            ((-3.0, 3.0),) * 3,
            192,
            is_comoving=True,
            claim="de Sitter background; averaged ANEC/AWEC satisfied at the "
            "matched speed v_s = H R, pointwise NEC/WEC violated at the wall "
            "(arXiv:2502.13153)",
            params={"v_s": 0.1, "R": 1.0, "sigma": 8.0, "H": 0.1},
            wall_radius=1.0,
            r_max=3.0,
            grid_center=lambda m: m.v_s / m.H,
        ),
        ConstructionSpec(
            "S-shell",
            lambda v: sshell_default(v_s=v),
            0.02,
            "v_s",
            ((-25.0, 25.0),) * 3,
            60,
            is_comoving=True,
            wall_radius=15.0,
            r_max=25.0,
            claim="source-first Class-I positive-density shell",
        ),
        ConstructionSpec(
            "T-shell",
            lambda v: tshell_default(v_0=v),
            0.1,
            "v_0",
            ((-25.0, 25.0),) * 3,
            60,
            is_comoving=False,
            wall_radius=15.0,
            r_max=25.0,
            claim="origin-static transport shell (matter tilt v_0)",
        ),
    ]
    return {s.name: s for s in specs}


# Matched panel: common shift kinematics.
#
# Fuchs is designed at v_s = 0.02 and Garattini pins v_s = H R, so the match is to
# them, on the only shared invariants: the dimensionless shift kinematics. The
# Fuchs sigmoid (R_1 = 10, R_2 = 20) crosses 10-90% at 12.790029 and 17.209971, so
# R_c = 15, W = 4.419943, W/R_c = 0.294663. MATCHED_SIGMA reproduces that width at
# R = R_c, solved from the exact finite-R profile since 2 atanh(0.8)/sigma is not
# valid at sigma*R ~ 1.
MATCHED_V_S = 0.02
MATCHED_R_C = 15.0
MATCHED_WIDTH = 4.419943
MATCHED_SIGMA = 0.497115373
MATCHED_R_MAX = 2.0 * MATCHED_R_C

# Matter content, background curvature and compactness are NOT matched: Fuchs
# carries a two-boundary shell, Garattini a Lambda R^2, and only Fuchs' stress is
# velocity-independent. Type fractions and miss rates are comparable under common
# sampling; raw stress severity and energy positivity are not.
MATCHING_CAVEAT = (
    "Common shift kinematics only (v_s, R_c, W/R_c, box, wall band). Matter "
    "content, background curvature and compactness are construction-specific "
    "and are NOT matched; see MATCHED_* in construction_adapter.py."
)


def matched_registry() -> dict[str, ConstructionSpec]:
    """The four panel constructions at common dimensionless shift kinematics."""
    v, R, s = MATCHED_V_S, MATCHED_R_C, MATCHED_SIGMA
    common = dict(
        speed_param="v_s",
        bounds=((-MATCHED_R_MAX, MATCHED_R_MAX),) * 3,
        grid_n=64,
        is_comoving=True,
        wall_radius=R,
        r_max=MATCHED_R_MAX,
    )
    specs = [
        ConstructionSpec(
            "Alcubierre",
            lambda vv: AlcubierreMetric(v_s=vv, R=R, sigma=s),
            v,
            claim="baseline; NEC/WEC violated for all observers",
            params={"v_s": v, "R": R, "sigma": s},
            **common,
        ),
        ConstructionSpec(
            "Rodal",
            lambda vv: RodalMetric(v_s=vv, R=R, sigma=s),
            v,
            claim="global Hawking-Ellis Type I; reduced (not eliminated) violations",
            params={"v_s": v, "R": R, "sigma": s},
            **common,
        ),
        ConstructionSpec(
            "Fuchs",
            lambda vv: fuchs_default(v_s=vv, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=6.668692),
            v,
            claim="constant-velocity shell satisfying all energy conditions (arXiv:2405.02709)",
            params={
                "v_s": v,
                "R_1": 10.0,
                "R_2": 20.0,
                "R_b": 1.0,
                "r_s_param": 6.668692,
                "kernel": "moving_average",
                "sigmoid": "published (Fuchs Eq. 31-32)",
            },
            **common,
        ),
        ConstructionSpec(
            "Garattini",
            lambda vv: GarattiniMetric(v_s=vv, R=R, sigma=s, H=vv / R),
            v,
            claim="de Sitter background; averaged ANEC/AWEC satisfied at the "
            "matched speed v_s = H R, pointwise NEC/WEC violated at the "
            "wall (arXiv:2502.13153)",
            params={"v_s": v, "R": R, "sigma": s, "H": v / R, "Lambda_R2": 3.0 * (v) ** 2},
            grid_center=lambda m: m.v_s / m.H,
            **common,
        ),
    ]
    return {sp.name: sp for sp in specs}


def wall_cells(spec: ConstructionSpec, speed: float | None = None, n: int | None = None) -> float:
    """Worst-case cells across the 10-90% wall, on the grid actually evaluated.

    Delegates to the single shared witness :func:`warpax.grids.wall_cells_on_axis`.

    The previous implementation counted *nodes* falling in a ``[0.05, 0.95]``
    band on a **uniform** line while the certification ran on the **graded**
    grid, summed the ``+x`` and ``-x`` crossings into one figure (so a reported
    "6" was about 3 per wall normal), and renormalised the shape function by its
    peak-to-peak range, which is not meaningful for a shell scalar, whose
    disconnected transitions the normalisation merges.
    """
    from ..grids import wall_cells_on_axis, wall_clustered

    metric = spec.metric(speed)
    n = spec.grid_n if n is None else n
    grid = wall_clustered(metric, list(spec.bounds), (n, n, n), a=spec.cluster_a)
    return float(wall_cells_on_axis(metric, grid.axes[0]).cells)


def is_resolved(
    spec: ConstructionSpec, speed: float | None = None, n: int | None = None
) -> tuple[bool, float]:
    cells = wall_cells(spec, speed=speed, n=n)
    return cells >= MIN_WALL_CELLS, cells
