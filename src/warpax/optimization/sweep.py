"""Parameter space sweep for transport characterization.

2D sweep over (compactness, thickness_ratio). At each grid point,
builds a T-shell/S-shell metric from the target density, evaluates
transport and diagnostics, then certifies EC admissibility.

Transport is characterized by two observables:
  - max|beta^x|: coordinate shift magnitude (gauge-dependent proxy)
  - delta_tau: null round-trip asymmetry (gauge-invariant)
"""
from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple


import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .basis import default_theta, unpack_theta
from .basis import coeffs_to_profiles_sshell, coeffs_to_profiles_tshell
from .ec_constraints import ec_feasibility_frame_free
from ..transport.diagnostics import null_round_trip_asymmetry


class SweepPoint(NamedTuple):
    """Result at a single (compactness, thickness) grid point.

    compactness : M / R_2 (dimensionless).
    thickness_ratio : (R_2 - R_1) / R_2.
    rho_max : peak density scale.
    transport : max|beta^x| (gauge-dependent shift proxy).
    transport_invariant : delta_tau (null round-trip asymmetry, gauge-invariant).
    ec_feasible : all NEC/WEC/DEC margins >= 0.
    worst_ec_margin : most negative observer-robust margin.
    constraint_residual : mean(eps_H^2 + eps_M^2) over probes.
    mass : M_ADM from the constraint solver.
    tidal : A_geo at r = R_1/2 (passenger cabin).
    """

    compactness: float
    thickness_ratio: float
    rho_max: float
    transport: float
    transport_invariant: float
    ec_feasible: bool
    worst_ec_margin: float
    constraint_residual: float
    mass: float
    tidal: float


class SweepResult(NamedTuple):
    """Full parameter sweep results.

    points : full-length list of SweepPoint-or-None, one entry per grid
        cell in row-major order: list position ``k`` is grid cell
        ``(k // Nt, k % Nt)``. ``None`` marks a not-yet-evaluated cell
        (partial checkpoints written while a parallel sweep is running).

        The list is kept at full length with explicit ``None`` holes
        (``null`` in the JSON sidecar) so position always encodes the
        grid cell.
    compactness_values : 1D array of sampled compactness values.
    thickness_values : 1D array of sampled thickness ratios.
    """

    points: list[SweepPoint | None]
    compactness_values: Float[Array, "Nc"]
    thickness_values: Float[Array, "Nt"]

    def to_grids(self) -> dict[str, np.ndarray]:
        """Reshape scalar fields to 2D grids (Nc, Nt) for plotting.

        Placement is by list position: ``points[k]`` fills cell
        ``(k // Nt, k % Nt)``. ``None`` entries leave NaN/False holes.
        """
        nc = self.compactness_values.shape[0]
        nt = self.thickness_values.shape[0]

        grids = {
            "transport": np.full((nc, nt), np.nan),
            "transport_invariant": np.full((nc, nt), np.nan),
            "ec_feasible": np.full((nc, nt), False),
            "worst_ec_margin": np.full((nc, nt), np.nan),
            "constraint_residual": np.full((nc, nt), np.nan),
            "mass": np.full((nc, nt), np.nan),
            "tidal": np.full((nc, nt), np.nan),
            "rho_max": np.full((nc, nt), np.nan),
        }

        for k, pt in enumerate(self.points):
            i = k // nt
            j = k % nt
            if i >= nc or j >= nt:
                break
            if pt is None:
                continue
            grids["transport"][i, j] = pt.transport
            grids["transport_invariant"][i, j] = pt.transport_invariant
            grids["ec_feasible"][i, j] = pt.ec_feasible
            grids["worst_ec_margin"][i, j] = pt.worst_ec_margin
            grids["constraint_residual"][i, j] = pt.constraint_residual
            grids["mass"][i, j] = pt.mass
            grids["tidal"][i, j] = pt.tidal
            grids["rho_max"][i, j] = pt.rho_max

        return grids

    def save(self, path: str) -> None:
        """Save to .npz + .json sidecar.

        Format: the JSON sidecar is a full-length list with one record
        per grid cell in row-major order; not-yet-evaluated cells are
        serialized as explicit ``null`` so list position always maps to
        grid cell ``(k // Nt, k % Nt)``.
        """
        grids = self.to_grids()
        np.savez(
            path,
            compactness_values=np.asarray(self.compactness_values),
            thickness_values=np.asarray(self.thickness_values),
            **grids,
        )
        json_path = str(path).replace(".npz", ".json")
        records = [None if pt is None else pt._asdict() for pt in self.points]
        for r in records:
            if r is not None:
                r["ec_feasible"] = bool(r["ec_feasible"])
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)

    @staticmethod
    def load(path: str) -> SweepResult:
        """Load from .npz + .json sidecar.

        ``null`` JSON records map back to ``None`` holes so positional
        grid-cell placement survives a save/load round trip.
        """
        data = np.load(path)
        compactness_values = jnp.asarray(data["compactness_values"])
        thickness_values = jnp.asarray(data["thickness_values"])

        json_path = str(path).replace(".npz", ".json")
        with open(json_path) as f:
            records = json.load(f)

        points = [None if r is None else SweepPoint(**r) for r in records]
        return SweepResult(
            points=points,
            compactness_values=compactness_values,
            thickness_values=thickness_values,
        )


def _rho_from_compactness(
    compactness: float,
    R_1: float,
    R_2: float,
) -> float:
    """Estimate rho_0 for target compactness M/R_2.

    M ~ (4pi/3)(R_2^3 - R_1^3) rho_0, so
    rho_0 = compactness * R_2 / ((4pi/3)(R_2^3 - R_1^3)).
    """
    shell_vol = (4.0 / 3.0) * np.pi * (R_2**3 - R_1**3)
    if shell_vol < 1e-30:
        return 1e-6
    return compactness * R_2 / shell_vol


def _evaluate_point(
    *,
    ansatz: str,
    R_1: float,
    R_2: float,
    rho_0: float,
    n_density: int,
    n_velocity: int,
    n_grid: int,
    n_probes: int,
    n_ec_starts: int,
) -> dict:
    """Build metric at a single design point and evaluate all diagnostics."""
    import jax

    theta = default_theta(n_density, n_velocity, v_0=0.1, rho_scale=rho_0)
    coeffs = unpack_theta(theta, n_density, n_velocity)

    if ansatz == "tshell":
        profiles = coeffs_to_profiles_tshell(coeffs, R_1, R_2)
        from ..metrics.tshell import tshell_from_profiles
        metric = tshell_from_profiles(profiles, n_grid=n_grid)
        transport = float(jnp.max(jnp.abs(metric._beta_x_grid)))
    else:
        profiles = coeffs_to_profiles_sshell(coeffs, R_1, R_2)
        from ..metrics.sshell import sshell_from_profiles
        metric = sshell_from_profiles(profiles, v_s=0.0, n_grid=n_grid)
        transport = 0.0

    # Null round-trip asymmetry (gauge-invariant transport observable).
    # Skipped for S-shell (zero shift) and tshell points with negligible
    # shift, to avoid the per-point cost of two null-geodesic integrations.
    transport_invariant = 0.0
    if ansatz == "tshell" and transport > 1e-6:
        emitter = jnp.array([0.0, R_1 * 0.5, 0.0, 0.0], dtype=jnp.float64)
        receiver = jnp.array([0.0, R_2 * 1.5, 0.0, 0.0], dtype=jnp.float64)
        try:
            transport_invariant = float(null_round_trip_asymmetry(
                metric, emitter, receiver,
                tau_max=R_2 * 5.0, num_points=200,
            ))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            warnings.warn(
                f"null_round_trip_asymmetry failed at "
                f"(R_1={R_1:.2f}, R_2={R_2:.2f}): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            transport_invariant = float("nan")

    mass = float(metric.total_mass)

    # Constraint residuals
    from ..constraints.residuals import normalized_residuals
    margin = 0.02 * (R_2 - R_1)
    r_probes = jnp.linspace(R_1 + margin, R_2 - margin, max(n_probes, 3))

    def eval_constraint(r_val):
        coords = jnp.array([0.0, r_val, 0.0, 0.0])
        res = normalized_residuals(metric, coords)
        return res["epsilon_H"]**2 + res["epsilon_M"]**2

    constraint_vals = jax.vmap(eval_constraint)(r_probes)
    constraint_residual = float(jnp.mean(constraint_vals))

    # Tidal force at passenger cabin
    from ..transport.diagnostics import geodesic_deviation_diagnostic
    tidal = float(geodesic_deviation_diagnostic(
        metric, jnp.array([0.0, R_1 * 0.5, 0.0, 0.0]),
    ))

    # EC certification: frame-free Hawking--Ellis verdict (cap-free Type-I
    # slacks + Type-IV detection), no observer search. Probes span the full
    # matter support including the smoothstep tails outside [R_1, R_2] where
    # the tilted shift drives the low-density edge Type-IV (criterion D).
    w = 0.05 * (R_2 - R_1)
    r_cert = jnp.linspace(R_1, R_2 + 3.0 * w, n_probes)
    ec_feas = ec_feasibility_frame_free(metric, r_cert)

    return {
        "rho_max": rho_0,
        "transport": transport,
        "transport_invariant": transport_invariant,
        "ec_feasible": ec_feas.feasible,
        "worst_ec_margin": ec_feas.worst_margin,
        "constraint_residual": constraint_residual,
        "mass": mass,
        "tidal": tidal,
    }


def sweep_transport(
    *,
    ansatz: str = "tshell",
    compactness_range: tuple[float, float] = (0.01, 0.20),
    thickness_range: tuple[float, float] = (0.3, 0.8),
    n_compactness: int = 12,
    n_thickness: int = 10,
    R_2: float = 20.0,
    n_density: int = 4,
    n_velocity: int = 4,
    n_grid: int = 512,
    n_probes: int = 15,
    n_ec_starts: int = 4,
    progress: bool = True,
    save_path: str | None = None,
    parallel: int | None = None,
) -> SweepResult:
    """Sweep (compactness, thickness_ratio) design space.

    At each grid point:
      1. Compute R_1, rho_0 from compactness and thickness.
      2. Build the metric with default Bernstein profiles.
      3. Read transport (max|beta^x|), constraint residuals, tidal force.
      4. Certify EC admissibility with the frame-free Hawking-Ellis certifier.

    Parameters
    ----------
    ansatz : "sshell" or "tshell".
    compactness_range : (min, max) for M/R_2.
    thickness_range : (min, max) for (R_2 - R_1) / R_2.
    n_compactness, n_thickness : grid resolution.
    R_2 : outer shell radius (scale parameter).
    n_density, n_velocity : Bernstein coefficient counts.
    n_grid : constraint solver resolution.
    n_probes : probe count for EC certification + constraint evaluation.
    n_ec_starts : BFGS multi-start for EC certification.
    progress : show tqdm progress bar.
    save_path : save intermediate results to this path.
    parallel : optional thread-pool worker count for grid-point evaluation.
        ``None`` (default) runs the serial loop. Setting an integer
        >= 2 dispatches grid
        points via :class:`concurrent.futures.ThreadPoolExecutor`; each
        thread calls into JIT'd JAX kernels which release the GIL, so
        Python dispatch overhead overlaps across points. Honors the
        ``WARPAX_SWEEP_PARALLEL`` env override when not set explicitly.
        Note: vmap/lax.map over the (compactness, thickness) grid is
        not currently feasible because ``_evaluate_point`` carries
        Python control flow (try/except on geodesic failures, branch
        on ansatz string, ec_feasibility_frame_free returns a Python dict).

    """
    if parallel is None:
        env_p = os.environ.get("WARPAX_SWEEP_PARALLEL")
        if env_p is not None:
            try:
                parallel = max(1, int(env_p))
            except ValueError:
                parallel = None

    c_vals = jnp.linspace(*compactness_range, n_compactness)
    t_vals = jnp.linspace(*thickness_range, n_thickness)

    total = n_compactness * n_thickness

    def _eval_idx(idx: int) -> SweepPoint:
        i = idx // n_thickness
        j = idx % n_thickness
        compactness = float(c_vals[i])
        thickness_ratio = float(t_vals[j])
        R_1 = R_2 * (1.0 - thickness_ratio)
        rho_0 = _rho_from_compactness(compactness, R_1, R_2)
        try:
            result = _evaluate_point(
                ansatz=ansatz,
                R_1=R_1,
                R_2=R_2,
                rho_0=rho_0,
                n_density=n_density,
                n_velocity=n_velocity,
                n_grid=n_grid,
                n_probes=n_probes,
                n_ec_starts=n_ec_starts,
            )
            return SweepPoint(
                compactness=compactness,
                thickness_ratio=thickness_ratio,
                **result,
            )
        except Exception as exc:
            warnings.warn(
                f"Grid point ({compactness:.4f}, {thickness_ratio:.3f}) "
                f"failed: {exc!r}",
                stacklevel=1,
            )
            return SweepPoint(
                compactness=compactness,
                thickness_ratio=thickness_ratio,
                rho_max=rho_0,
                transport=0.0,
                transport_invariant=0.0,
                ec_feasible=False,
                worst_ec_margin=-1.0,
                constraint_residual=float("nan"),
                mass=float("nan"),
                tidal=float("nan"),
            )

    points: list[SweepPoint | None] = [None] * total

    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="Sweep", unit="pt")
        except ImportError:
            pbar = None
    else:
        pbar = None

    def _finalize_idx(idx: int, pt: SweepPoint) -> None:
        points[idx] = pt
        if pbar is not None:
            pbar.update(1)
        if save_path is not None and (idx + 1) % max(1, total // 10) == 0:
            # Checkpoint the FULL-length list (None holes included);
            # compacting it scrambles grid-cell placement under parallel
            # completion order.
            partial = SweepResult(
                points=list(points),
                compactness_values=c_vals,
                thickness_values=t_vals,
            )
            partial.save(save_path)

    try:
        if parallel is not None and parallel >= 2:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(_eval_idx, idx): idx for idx in range(total)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    _finalize_idx(idx, fut.result())
        else:
            for idx in range(total):
                _finalize_idx(idx, _eval_idx(idx))
    finally:
        if pbar is not None:
            pbar.close()

    # All cells are filled at this point (failures yield sentinel points);
    # keep the full-length positional list for index -> cell placement.
    result = SweepResult(
        points=list(points),
        compactness_values=c_vals,
        thickness_values=t_vals,
    )

    if save_path is not None:
        result.save(save_path)

    return result


