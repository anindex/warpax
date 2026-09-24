"""Elastic-shell equilibrium, frame dragging, centrifugal tides and clock shifts.

Run with `python examples/11_elastic_shell.py` after installing the solver
extra. Profiles and numerical comparisons are written to results/elastic_shell.
The exact response coefficients in that directory are input data from the
six free-traction equations. No sibling manuscript checkout is required.
"""

import csv
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.special import expit

OUT = Path(__file__).resolve().parents[1] / "results" / "elastic_shell"
OUT.mkdir(parents=True, exist_ok=True)


def mass(r):
    t = np.clip((np.asarray(r) - 10) / 10, 0, 1)
    return 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7


def dm(r):
    t = np.clip((np.asarray(r) - 10) / 10, 0, 1)
    return 14 * t**3 * (1 - t) ** 3


def density(r):
    return dm(r) / (4 * np.pi * np.asarray(r) ** 2)


def isotropic(tol):
    def rhs(r, y):
        p = y[0]
        m = mass(r)
        return [-(density(r) + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))]

    sol = solve_ivp(rhs, (20, 10), [0.0], rtol=tol, atol=tol * 1e-8, dense_output=True)
    assert sol.success, sol.message
    assert sol.y[0, -1] > 0, "Inner boundary pressure must be positive"
    return float(sol.y[0, -1])


# The exact anisotropic counterexample has M=1, R1=10, R2=20.
r = np.linspace(10, 20, 1001)
m = mass(r)
rho = density(r)
pt = rho * m / (2 * (r - 2 * m))
speed2 = m / (r - 2 * m)
assert np.all(rho >= 0) and np.all(pt >= 0) and np.all(pt <= rho)
assert np.max(6 * m / r) < 1
lapse_inner = np.exp(
    np.log(1 - 2 / 20) / 2
    - quad(lambda x: mass(x) / (x * (x - 2 * mass(x))), 10, 20, epsabs=1e-13)[0]
)
with (OUT / "static_counterexample.csv").open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["r", "m", "rho", "p_r", "p_t", "DEC_slack", "circular_speed_squared"])
    w.writerows(zip(r, m, rho, np.zeros_like(r), pt, rho - pt, speed2, strict=True))

# Full, finite-strain quasi-Hookean energy with a relaxed compressional state.
# Material radius R is the independent variable. a=n_r, b=n_t=R/r.
a, b = sp.symbols("a b", positive=True)
m0 = 1e-6
bulk = 0.2 * m0
shear = 0.1 * m0
n = a * b * b
s2 = (a / b - b / a) ** 2 / 6
energy = m0 * n + bulk * (n - 1) ** 2 / 2 + shear * n * s2
pr_expr = a * sp.diff(energy, a) - energy
pt_expr = b * sp.diff(energy, b) / 2 - energy
constitutive = sp.lambdify(
    (a, b), [energy, pr_expr, pt_expr, sp.diff(pr_expr, a), sp.diff(pr_expr, b)], "numpy", cse=True
)


def initial(b0):
    a0 = brentq(lambda x: constitutive(x, b0)[1], 0.1, 4, xtol=1e-14)
    return [10 / b0, 0, a0, 0]


def rhs(R, y):
    rad, massval, nr, phi = y
    nt = R / rad
    F = 1 - 2 * massval / rad
    if F <= 0 or nr <= 0 or nt <= 0:
        raise ValueError("Exited horizon-free elastic domain")
    rho, pr, pt, pa, pb = constitutive(nr, nt)
    dr = np.sqrt(F) / nr
    db = 1 / rad - nt * dr / rad
    dphi = (massval + 4 * np.pi * rad**3 * pr) / (rad * (rad - 2 * massval))
    force = -(rho + pr) * dphi + 2 * (pt - pr) / rad
    return [dr, 4 * np.pi * rad**2 * rho * dr, (force * dr - pb * db) / pa, dphi * dr]


def elastic(tol, step=0.1):
    def integrate(b0, dense=False):
        sol = solve_ivp(
            rhs, (10, 20), initial(b0), rtol=tol, atol=tol * 1e-3, dense_output=dense, max_step=step
        )
        assert sol.success, sol.message
        return sol

    def outer(b0):
        sol = integrate(b0)
        rad, _, nr, _ = sol.y[:, -1]
        return constitutive(nr, 20 / rad)[1] / m0

    b0 = brentq(outer, 0.95, 1.05, xtol=1e-13)
    sol = integrate(b0, True)
    R = np.linspace(10, 20, 1001)
    rad, massval, nr, phi = sol.sol(R)
    nt = R / rad
    rho, pr, pt, pa, pb = constitutive(nr, nt)
    assert np.all(rho - np.maximum(abs(pr), abs(pt)) > 0)
    assert abs(pr[0]) < 1e-16 and abs(pr[-1]) < 1e-14
    # Longitudinal principal speed: n_i dp_i/dn_i /(rho+p_i).
    cr2 = nr * pa / (rho + pr)
    assert np.all((cr2 > 0) & (cr2 < 1))
    assert np.all((nr > 0.99) & (nr < 1.01) & (nt > 0.99) & (nt < 1.01))
    return {
        "rtol": tol,
        "max_step": step,
        "inner_radius": float(rad[0]),
        "outer_radius": float(rad[-1]),
        "mass": float(massval[-1]),
        "inner_nt": b0,
        "outer_traction": float(pr[-1]),
        "min_DEC": float(np.min(rho - np.maximum(abs(pr), abs(pt)))),
        "max_compactness": float(np.max(2 * massval / rad)),
        "radial_longitudinal_speed_squared_range": [float(cr2.min()), float(cr2.max())],
    }, (R, rad, massval, nr, nt, rho, pr, pt, phi)


def compatible_radial_data(static, amplitude):
    """Numerical check of the static-collar construction; not an interval proof."""
    outer = static.sol(20)
    old_mass = outer[1]

    def solve(candidate):
        terminal = outer.copy()
        terminal[1] = candidate
        collar = solve_ivp(
            rhs, (20, 18), terminal, rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.05
        )
        assert collar.success, collar.message

        def geometry(R):
            base = static.sol(R)
            slope = rhs(R, base)[0]
            weight = dweight = 0.0
            if R >= 19:
                weight = 1.0
            elif R > 18:
                s = R - 18
                weight = expit(1 / (1 - s) - 1 / s)
                dweight = weight * (1 - weight) * (1 / s**2 + 1 / (1 - s) ** 2)
            rad = base[0]
            if weight:
                local = collar.sol(R)
                local_slope = rhs(R, local)[0]
                rad += weight * (local[0] - base[0])
                slope += weight * (local_slope - slope) + dweight * (local[0] - base[0])
            s = (R - 13) / 3
            velocity = amplitude * np.exp(4 - 1 / (s * (1 - s))) if 0 < s < 1 else 0.0
            return rad, slope, velocity

        def constraint(R, m):
            rad, slope, velocity = geometry(R)
            gamma2 = 1 + velocity**2 - 2 * m[0] / rad
            assert gamma2 > 0 and slope > 0
            nr = np.sqrt(gamma2) / slope
            rho = constitutive(nr, R / rad)[0]
            return [4 * np.pi * rad**2 * slope * rho]

        sol = solve_ivp(
            constraint, (10, 20), [0], rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.05
        )
        assert sol.success, sol.message
        return sol.y[0, -1] - candidate, sol, collar, geometry

    bracket = max(1e-8, old_mass * amplitude**2)
    root = brentq(lambda mass: solve(mass)[0], old_mass - bracket, old_mass + bracket, xtol=1e-14)
    residual, sol, collar, geometry = solve(root)
    errors = []
    tractions = []
    min_dec = float("inf")
    max_velocity = 0.0
    for R in np.linspace(10, 20, 201):
        rad, slope, velocity = geometry(R)
        mass = sol.sol(R)[0]
        nr = np.sqrt(1 + velocity**2 - 2 * mass / rad) / slope
        rho, pr, pt, *_ = constitutive(nr, R / rad)
        min_dec = min(min_dec, rho - max(abs(pr), abs(pt)))
        max_velocity = max(max_velocity, abs(velocity))
        if R >= 19:
            errors.append(abs(mass - collar.sol(R)[1]))
        if R in (10, 20):
            tractions.append(float(pr))
    assert abs(residual) < 1e-12 and max(errors) < 1e-10
    assert max(abs(p) for p in tractions) < 1e-14 and min_dec > 0
    assert root > old_mass and max_velocity > 0
    return {
        "amplitude": amplitude,
        "mass": float(root),
        "mass_increase": float(root - old_mass),
        "mass_increase_over_amplitude_squared": float((root - old_mass) / amplitude**2),
        "endpoint_residual": float(residual),
        "outer_collar_mass_error": float(max(errors)),
        "face_tractions": tractions,
        "sampled_min_DEC": float(min_dec),
        "scope": (
            "Numerical construction of nonzero radial data with static collars and a solved "
            "mass constraint. Smoothness and all-order compatibility are written arguments; "
            "this is not an evolution or interval certificate."
        ),
    }


def centrifugal_collocation():
    """Integrate the radial Lamé equations independently of the power-law solve."""
    from fractions import Fraction

    cases = json.loads((OUT / "centrifugal_response.json").read_text())["cases"]
    errors = []
    for row in cases:
        beta, bulk, mu = [float(Fraction(row[k])) for k in ("beta", "bulk_ratio", "shear_ratio")]
        lam = bulk - 2 * mu / 3
        load = 1 + lam - mu

        def rhs(s, y):
            U0, dU0, U, dU, V, dV = y
            D = dU + 2 * U / s - 6 * V / s
            ddU0 = -2 * dU0 / s + 2 * U0 / s**2 - 2 * load * s / (3 * (lam + 2 * mu))
            ddU = (
                2 * load * s / 3
                - mu * (2 * dU / s - 8 * U / s**2 + 12 * V / s**2)
                - (lam + mu) * (2 * dU / s - 2 * U / s**2 - 6 * dV / s + 6 * V / s**2)
            ) / (lam + 2 * mu)
            ddV = (
                -2 * dV / s + 6 * V / s**2 - 2 * U / s**2 + (load * s / 3 - (lam + mu) * D / s) / mu
            )
            return np.array([dU0, ddU0, dU, ddU, dV, ddV])

        def bc(left, right):
            residual = []
            for s, y in ((1, left), (beta, right)):
                U0, dU0, U, dU, V, dV = y
                residual.extend(
                    [
                        lam * (dU0 + 2 * U0 / s) + 2 * mu * dU0 + lam * s * s / 3,
                        lam * (dU + 2 * U / s - 6 * V / s) + 2 * mu * dU - lam * s * s / 3,
                        dV + (U - V) / s,
                    ]
                )
            return np.array(residual)

        mesh = np.linspace(1, beta, 81)
        solution = solve_bvp(rhs, bc, mesh, np.zeros((6, len(mesh))), tol=1e-8, max_nodes=3000)
        assert solution.success, solution.message
        s = np.linspace(1, beta, 201)
        h, j, a, b, c, d = [float(Fraction(x)) for x in row["coefficients"]]
        exact = np.array(
            [
                h * s + j / s**2 - load * s**3 / (15 * (lam + 2 * mu)),
                2 * a * s
                + (6 * lam * b + load / 3) * s**3 / (5 * lam + 7 * mu)
                + (3 * lam + 5 * mu) * c / (mu * s**2)
                - 3 * d / s**4,
                a * s + b * s**3 + c / s**2 + d / s**4,
            ]
        )
        err = float(np.max(abs(solution.sol(s)[[0, 2, 4]] - exact)) / np.max(abs(exact)))
        assert err < 1e-8, err

        def integrand(s):
            _, _, U, dU, V, _ = solution.sol(s)
            D = dU + 2 * U / s - 6 * V / s
            return (-3 * U + 6 * V) / s**2 - 3 * bulk * D / s - (1 - bulk) * s

        measured = 4 / 5 * quad(integrand, 1, beta, epsabs=1e-10, epsrel=1e-10)[0]
        expected = float(Fraction(row["minus_q_over_pi_G_m0_A2_Omega2"]))
        moment_error = abs(measured - expected) / abs(expected)
        assert moment_error < 1e-8, moment_error

        def clock_integrand(s):
            U0, dU0 = solution.sol(s)[:2]
            return U0 + 3 * bulk * s * (dU0 + 2 * U0 / s) - (1 - bulk) * s**3

        clock_measured = 4 * quad(clock_integrand, 1, beta, epsabs=1e-10, epsrel=1e-10)[0]
        clock_expected = float(Fraction(row["clock_shift_over_pi_chi_epsilon"]))
        clock_error = abs(clock_measured - clock_expected) / max(1, abs(clock_expected))
        assert clock_error < 1e-8, clock_error
        errors.append(
            {
                "beta": beta,
                "bulk_ratio": bulk,
                "shear_ratio": mu,
                "profile_relative_error": err,
                "moment_relative_error": moment_error,
                "clock_scaled_error": clock_error,
            }
        )
    return {
        "scope": (
            "Independent collocation at 28 discrete parameter triples, including the tidal "
            "sign reversal; leading coefficients only."
        ),
        "cases": errors,
    }


if __name__ == "__main__":
    res = {
        "isotropic_inner_pressure": [
            {"rtol": tol, "p_inner": isotropic(tol)} for tol in (1e-8, 1e-10, 1e-12)
        ],
        "cluster": {
            "mass": 1,
            "R1": 10,
            "R2": 20,
            "cavity_lapse": float(lapse_inner),
            "max_2m_over_r": float(np.max(2 * m / r)),
            "max_pt_over_rho": float(np.max(m / (2 * (r - 2 * m)))),
        },
        "elastic": [],
    }
    for tol, step in ((1e-7, 1.0), (1e-9, 0.5), (1e-11, 0.1)):
        result, values = elastic(tol, step)
        res["elastic"].append(result)
    # Independent collocation discretization, including the lapse normalization.
    R, rad, massval, nr, nt, rho, pr, pt, phi = values
    phi = phi - phi[-1] + np.log(1 - 2 * massval[-1] / rad[-1]) / 2

    def vector_rhs(x, y):
        return np.asarray([rhs(xx, yy) for xx, yy in zip(x, y.T, strict=True)]).T

    def bc(y0, y1):
        return [
            y0[1],
            constitutive(y0[2], 10 / y0[0])[1] / m0,
            constitutive(y1[2], 20 / y1[0])[1] / m0,
            y1[3] - np.log(1 - 2 * y1[1] / y1[0]) / 2,
        ]

    coll = solve_bvp(
        vector_rhs,
        bc,
        R[::20],
        np.array([rad, massval, nr, phi])[:, ::20],
        tol=1e-10,
        max_nodes=1000,
    )
    assert coll.success, coll.message
    err = float(np.max(np.abs(coll.sol(R) - np.array([rad, massval, nr, phi]))))
    assert err < 1e-7
    res["elastic_collocation"] = {
        "max_state_difference": err,
        "max_rms_residual": float(np.max(coll.rms_residuals)),
        "boundary_residuals": bc(coll.y[:, 0], coll.y[:, -1]),
    }

    # Separate integrals of local energy and redshifted active source.
    def mass_integrands(R):
        r, m, nr, phi = coll.sol(R)
        rho, pr, pt, *_ = constitutive(nr, R / r)
        F = 1 - 2 * m / r
        dr = np.sqrt(F) / nr
        return (
            4 * np.pi * r * r * (rho + pr + 2 * pt) * np.exp(phi) / np.sqrt(F) * dr,
            4 * np.pi * r * r * rho / np.sqrt(F) * dr,
            (m + 4 * np.pi * r**3 * pr) / (r * r * F) * dr,
        )

    komar, proper, lapse_drop = [
        quad(lambda R: mass_integrands(R)[i], 10, 20, epsabs=1e-13, epsrel=1e-11)[0]
        for i in range(3)
    ]
    outer_mass = float(coll.sol(20)[1])
    reference = 4 * np.pi * m0 * (20**3 - 10**3) / 3
    binding = 16 * np.pi**2 * m0**2 / 3 * ((20**5 - 10**5) / 5 - 10**3 * (20**2 - 10**2) / 2)
    clock = np.sqrt(1 - 2 * outer_mass / coll.sol(20)[0]) * np.exp(-lapse_drop)
    assert abs(komar / outer_mass - 1) < 1e-8
    assert proper > reference > outer_mass
    assert abs(clock / np.exp(coll.sol(10)[3]) - 1) < 1e-10
    res["mass_and_clock"] = {
        "ADM_mass": outer_mass,
        "Komar_mass": komar,
        "proper_material_energy": proper,
        "reference_rest_energy": reference,
        "binding_to_reference": reference - outer_mass,
        "leading_weak_binding": binding,
        "cavity_lapse": float(clock),
        "Komar_relative_error": abs(komar / outer_mass - 1),
        "scope": (
            "Independent quadratures on the collocation solution; no interval remainder "
            "for the weak-coupling expansion."
        ),
    }

    # First-order rotational response on the independently normalized static BVP.
    def dragging_rhs(R, y):
        rad, mass, nr, _ = coll.sol(R)
        rho, pr, pt, *_ = constitutive(nr, R / rad)
        F = 1 - 2 * mass / rad
        slope = np.sqrt(F) / nr
        f, k = y  # k = df/dr, even though the integration variable is R.
        return [
            k * slope,
            ((4 * np.pi * rad * (rho + pr) / F - 4 / rad) * k + 16 * np.pi * (rho + pt) * f / F)
            * slope,
        ]

    rotation = solve_ivp(
        dragging_rhs, (10, 20), [1, 0], rtol=1e-12, atol=1e-14, max_step=0.05, dense_output=True
    )
    assert rotation.success, rotation.message
    f, k = rotation.sol(R)
    inner, outer = rad[0], rad[-1]
    normalizer = f[-1] + outer * k[-1] / 3
    omega = 1 - f / normalizer
    inertia = outer**4 * k[-1] / (6 * normalizer)
    assert np.all((omega > 0) & (omega < 1)) and np.all(np.diff(omega) <= 1e-14)
    assert abs(omega[-1] - 2 * inertia / outer**3) < 1e-13
    assert abs(-k[-1] / normalizer + 6 * inertia / outer**4) < 1e-13
    assert k[0] == 0
    vacuum_inner = np.linspace(0, inner, 101, endpoint=False)
    vacuum_outer = np.linspace(outer, 3 * outer, 301)[1:]
    with (OUT / "elastic_dragging.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["r", "omega_over_varpi", "domega_dr_over_varpi", "region"])
        writer.writerows(
            zip(
                vacuum_inner,
                np.full_like(vacuum_inner, omega[0]),
                np.zeros_like(vacuum_inner),
                np.zeros_like(vacuum_inner, dtype=int),
                strict=True,
            )
        )
        writer.writerows(
            zip(rad, omega, -k / normalizer, np.ones_like(rad, dtype=int), strict=True)
        )
        writer.writerows(
            zip(
                vacuum_outer,
                2 * inertia / vacuum_outer**3,
                -6 * inertia / vacuum_outer**4,
                np.full_like(vacuum_outer, 2, dtype=int),
                strict=True,
            )
        )
    res["linear_rotation"] = {
        "cavity_dragging_over_varpi": float(omega[0]),
        "angular_momentum_over_varpi": float(inertia),
        "scope": "First-order response about the spherical BVP; no finite-rotation remainder.",
    }
    redshift_leading = 2 * np.pi * m0 * (20**2 - 10**2)
    dragging_leading = 4 * redshift_leading / 3
    assert abs((1 - clock) / redshift_leading - 1) < 0.002
    assert abs(omega[0] / dragging_leading - 1) < 0.002
    res["weak_cavity_comparison"] = {
        "leading_redshift": redshift_leading,
        "leading_dragging": dragging_leading,
        "dragging_over_redshift": float(omega[0] / (1 - clock)),
        "redshift_relative_correction": float((1 - clock) / redshift_leading - 1),
        "dragging_relative_correction": float(omega[0] / dragging_leading - 1),
        "scope": "Comparison at chi=1e-4, beta=2; not a bound on the asymptotic remainder.",
    }
    # Independent endpoint-mass variation, with both free faces displaced.
    # Compare sensitivity ODEs to the derived radial quadratic form at G=1.
    Rv, rv, sv, Vv, Mv = sp.symbols("R r slope V M", positive=True)
    mass_rhs = (
        4 * sp.pi * rv**2 * sv * energy.subs({a: sp.sqrt(1 + Vv**2 - 2 * Mv / rv) / sv, b: Rv / rv})
    )
    variables = (rv, sv, Vv, Mv)
    gradient = sp.lambdify(
        (Rv, *variables), [sp.diff(mass_rhs, x).subs(Vv, 0) for x in variables], "numpy", cse=True
    )
    hessian = sp.lambdify(
        (Rv, *variables), sp.hessian(mass_rhs, variables).subs(Vv, 0), "numpy", cse=True
    )
    moduli = sp.lambdify(
        (a, b),
        [a * sp.diff(pr_expr, a), b * sp.diff(pr_expr, b), b * sp.diff(pt_expr, b)],
        "numpy",
        cse=True,
    )
    # Sampled radial diagnostic, kept separate from the continuum enclosure.
    Am, Bm, Dm = moduli(nr, nt)
    F = 1 - 2 * massval / rad
    phip = (massval + 4 * np.pi * rad**3 * pr) / (rad**2 * F)
    q = pt - pr
    enthalpy = rho + pr
    restoring = (
        (2 * q + 2 * Dm - Bm**2 / Am) / rad**2
        + enthalpy * (8 * np.pi * pr / F - phip**2 - 4 * phip / rad)
        - 4 * q * phip / rad
    )
    sampled_gap = np.exp(2 * phi) * F * restoring / enthalpy
    assert np.min(sampled_gap) > 0
    res["radial_diagnostic"] = {
        "sampled_minimum_frequency_bound": float(np.min(sampled_gap)),
        "scope": "Sampled sufficient radial criterion; not a continuum stability bound.",
    }

    def mass_variation(R, y):
        rad, m, nr, phi = coll.sol(R)
        nt = R / rad
        F = 1 - 2 * m / rad
        slope = np.sqrt(F) / nr
        lam = -np.log(F) / 2
        density, pressure, tangent, *_ = constitutive(nr, nt)
        w = density + pressure
        q = tangent - pressure
        Am, Bm, Dm = moduli(nr, nt)
        phip = (m + 4 * np.pi * rad**3 * pressure) / (rad**2 * F)
        restoring = (
            (2 * q + 2 * Dm - Bm**2 / Am) / rad**2
            + w * (8 * np.pi * pressure / F - phip**2 - 4 * phip / rad)
            - 4 * q * phip / rad
        )
        u = 0.3 + 0.02 * (R - 10) + 0.05 * np.sin(np.pi * (R - 10) / 10)
        du = 0.02 + 0.005 * np.pi * np.cos(np.pi * (R - 10) / 10)
        v = 0.07 * np.cos(np.pi * (R - 10) / 10)
        direction = np.array([u, du, v, y[0]])
        args = (R, rad, slope, 0, m)
        grad = np.asarray(gradient(*args))
        hes = np.asarray(hessian(*args))
        predicted = (
            4
            * np.pi
            * rad**2
            * np.exp(lam + phi)
            * slope
            * (
                Am * (du / slope - phip * u + Bm * u / (Am * rad)) ** 2
                + restoring * u**2
                + w * v**2 / F
            )
        )
        return [grad @ direction, grad[3] * y[1] + direction @ hes @ direction, predicted]

    variation = solve_ivp(mass_variation, (10, 20), [0, 0, 0], rtol=1e-11, atol=1e-14, max_step=0.1)
    assert variation.success, variation.message
    first, second, predicted = variation.y[:, -1]
    relative_error = abs(second - predicted) / abs(predicted)
    assert abs(first) < 1e-11 and predicted > 0 and relative_error < 1e-8
    res["constrained_mass_hessian"] = {
        "first_variation": float(first),
        "endpoint_hessian": float(second),
        "radial_form": float(predicted),
        "relative_difference": float(relative_error),
        "scope": "Numerical cross-check of the finite-gravity mass Hessian; proof is in the appendix.",
    }
    res["compatible_radial_data"] = [
        compatible_radial_data(coll, eps) for eps in (0.02, 0.01, 0.005)
    ]
    kinetic_ratios = [
        row["mass_increase_over_amplitude_squared"] for row in res["compatible_radial_data"]
    ]
    assert max(kinetic_ratios) / min(kinetic_ratios) < 1.001
    with (OUT / "elastic_static.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["R", "r", "m", "n_r", "n_t", "rho", "p_r", "p_t", "Phi_unnormalized"])
        w.writerows(zip(*values, strict=True))
    res["centrifugal_collocation"] = centrifugal_collocation()
    (OUT / "numerics.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
