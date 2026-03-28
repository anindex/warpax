"""test_v1_api_surface - pin public API surface (frozen 2026-04-18).

Discipline:
  - The fixture `tests/fixtures/v1_api_surface_v1_0.json` is the frozen
    public API surface as of the 2026-04-18 baseline commit.
  - The fixture `tests/fixtures/v1_api_defaults_v1_0.json` is the frozen
    default-parameter snapshot.
  - Any subsequent commit MUST keep every fixture symbol importable from its
    documented module AND every default value bit-equal to the fixture.
    Renames or removals require a `warnings.deprecated` alias kept under
    the OLD name for at least one minor version.
  - Additions to the surface or defaults are PERMITTED but require
    regenerating both fixtures (`pytest tests/test_v1_api_surface.py --regenerate`)
    AND a CHANGELOG entry.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SURFACE_FIXTURE_PATH = FIXTURE_DIR / "v1_api_surface_v1_0.json"
DEFAULTS_FIXTURE_PATH = FIXTURE_DIR / "v1_api_defaults_v1_0.json"


def _current_surface() -> dict[str, list[str]]:
    """Introspect every module named in the surface fixture and return its current `__all__`."""
    with open(SURFACE_FIXTURE_PATH) as fh:
        fixture = json.load(fh)
    surface: dict[str, list[str]] = {}
    for module_path in sorted(fixture.keys()):
        mod = importlib.import_module(module_path)
        symbols = sorted(getattr(mod, "__all__", []))
        surface[module_path] = symbols
    return surface


def _capture_defaults(surface: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """For every callable in `surface`, capture {param_name: repr(default)} for kwargs with defaults."""
    defaults: dict[str, dict[str, str]] = {}
    for module_path, symbols in surface.items():
        mod = importlib.import_module(module_path)
        for sym in symbols:
            if not hasattr(mod, sym):
                continue
            obj = getattr(mod, sym)
            # Classes: introspect __init__; functions: introspect directly.
            if inspect.isclass(obj):
                target = getattr(obj, "__init__", None)
                entry_key = f"{module_path}.{sym}.__init__"
            elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
                target = obj
                entry_key = f"{module_path}.{sym}"
            else:
                continue  # constants, module-level dicts, etc.
            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            params: dict[str, str] = {}
            for name, p in sig.parameters.items():
                if p.default is inspect.Parameter.empty:
                    continue
                params[name] = repr(p.default)
            if params:
                defaults[entry_key] = params
    return defaults


def _load_surface_fixture() -> dict[str, list[str]]:
    with open(SURFACE_FIXTURE_PATH) as fh:
        return json.load(fh)


def _load_defaults_fixture() -> dict[str, dict[str, str]]:
    with open(DEFAULTS_FIXTURE_PATH) as fh:
        return json.load(fh)


def test_fixture_exists_and_well_formed():
    assert SURFACE_FIXTURE_PATH.exists(), (
        f"v0.1.x API surface fixture missing at {SURFACE_FIXTURE_PATH}. "
        f"Run `pytest tests/test_v1_api_surface.py --regenerate` to seed."
    )
    assert DEFAULTS_FIXTURE_PATH.exists(), (
        f"v0.1.x API defaults fixture missing at {DEFAULTS_FIXTURE_PATH} (W1). "
        f"Run `pytest tests/test_v1_api_surface.py --regenerate` to seed."
    )
    fixture = _load_surface_fixture()
    assert "warpax" in fixture
    for mod, syms in fixture.items():
        assert isinstance(syms, list)
        assert syms == sorted(syms), f"{mod}: surface symbols must be alphabetically sorted"


def test_v1_api_surface_no_removed_symbols():
    """Every fixture symbol MUST still be importable from its documented module.

    A change renaming `verify_grid` to `verify_curvature_grid` (without
    keeping a `verify_grid` deprecated alias) breaks existing users - this
    test catches it.
    """
    fixture = _load_surface_fixture()
    missing: list[str] = []
    for module_path, symbols in fixture.items():
        mod = importlib.import_module(module_path)
        for sym in symbols:
            if not hasattr(mod, sym):
                missing.append(f"{module_path}.{sym}")
    assert not missing, (
        "v0.1.x API surface broken - the following symbols are no longer importable:\n"
        " - " + "\n - ".join(missing) + "\n\n"
        "If this is intentional (rename/removal), add a `@warnings.deprecated` "
        "(or `typing_extensions.deprecated`) alias under the OLD name and re-export it "
        "from the OLD module's `__all__`. Then this test passes again. "
        "See project conventions."
    )


def test_v1_api_surface_no_removed_modules():
    """Every fixture module must still be importable."""
    fixture = _load_surface_fixture()
    unimportable: list[str] = []
    for module_path in fixture.keys():
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            unimportable.append(f"{module_path}: {exc}")
    assert not unimportable, (
        "v0.1.x modules no longer importable:\n - " + "\n - ".join(unimportable)
    )


def test_v1_api_surface_subset_or_extension():
    """Current `__all__` MUST be a (non-strict) superset of the surface fixture per module.

    Additions are PERMITTED (and require fixture regen + CHANGELOG entry).
    Removals are NOT permitted (covered by `test_no_removed_symbols`).
    Reorderings are irrelevant (we sort).
    """
    fixture = _load_surface_fixture()
    current = _current_surface()
    regressions: list[str] = []
    for mod, fixture_syms in fixture.items():
        current_syms = set(current.get(mod, []))
        for sym in fixture_syms:
            if sym not in current_syms:
                regressions.append(f"{mod}.{sym} dropped from __all__")
    assert not regressions, (
        "v0.1.x API surface regressed (symbols dropped from `__all__`):\n - "
        + "\n - ".join(regressions)
    )


def test_v1_api_defaults():
    """Every callable's default-parameter values MUST be bit-equal to the
    defaults fixture.

    A flip from `n_starts=16` to `n_starts=8` (silent-default change) is a
    breaking change; this test makes it a CI failure.
    """
    defaults_fixture = _load_defaults_fixture()
    surface_fixture = _load_surface_fixture()
    current_defaults = _capture_defaults(surface_fixture)
    regressions: list[str] = []
    for entry_key, fixture_params in defaults_fixture.items():
        current_params = current_defaults.get(entry_key, {})
        for param_name, fixture_default_repr in fixture_params.items():
            current_default_repr = current_params.get(param_name, "<missing>")
            if current_default_repr != fixture_default_repr:
                regressions.append(
                    f"{entry_key}({param_name}=...): "
                    f"v0.1.x default was {fixture_default_repr}, "
                    f"current default is {current_default_repr}"
                )
    assert not regressions, (
        "v0.1.x default-parameter values flipped (violation):\n - "
        + "\n - ".join(regressions) + "\n\n"
        "v0.1.x default values are part of the frozen public API; bumping the "
        "major version is required to change them. If intentional for v2.0, "
        "regenerate the fixture AND bump major version."
    )


# Regeneration trigger: use an opt-in environment variable so the regen
# entry point lives entirely inside this test module:
#
# WARPAX_REGENERATE_API_FIXTURES=1 \
# python -m pytest tests/test_v1_api_surface.py::test_regenerate_fixture -v
#
# Without the env var the test SKIPS (no-op).
#
# WARPAX_REGENERATE_API_FIXTURES=1 \
# python -m pytest tests/test_v1_api_surface.py::test_regenerate_fixture -v
#
# Without the env var the test SKIPS (no-op; not a contract).
_REGEN_ENV_VAR = "WARPAX_REGENERATE_API_FIXTURES"


def test_regenerate_fixture(tmp_path):
    """When `WARPAX_REGENERATE_API_FIXTURES=1` is set, rewrite BOTH fixtures
    from the current tree. Without it this test SKIPS (no-op; not a contract).
    """
    if os.environ.get(_REGEN_ENV_VAR) != "1":
        pytest.skip(
            f"Set {_REGEN_ENV_VAR}=1 to refresh the v0.1.x API surface + defaults fixtures."
        )
    # Walk the candidate v0.1.x modules; include only those that exist.
    candidates = [
        "warpax",
        "warpax.analysis", "warpax.benchmarks", "warpax.energy_conditions",
        "warpax.geodesics", "warpax.geometry", "warpax.metrics",
        "warpax.visualization",
        # -or-later (skipped if not yet present):
        "warpax.io", "warpax.grids", "warpax.classify", "warpax.junction",
        "warpax.quantum", "warpax.averaged", "warpax.design",
    ]
    surface: dict[str, list[str]] = {}
    for mp in candidates:
        try:
            mod = importlib.import_module(mp)
        except ImportError:
            continue
        surface[mp] = sorted(getattr(mod, "__all__", []))
    SURFACE_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SURFACE_FIXTURE_PATH.write_text(
        json.dumps(surface, indent=2, sort_keys=True) + "\n"
    )
    # also regenerate the defaults fixture
    defaults = _capture_defaults(surface)
    DEFAULTS_FIXTURE_PATH.write_text(
        json.dumps(defaults, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\nRegenerated {SURFACE_FIXTURE_PATH} ({len(surface)} modules) "
        f"and {DEFAULTS_FIXTURE_PATH} ({len(defaults)} callables).",
        file=sys.stderr,
    )
