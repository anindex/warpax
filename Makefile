.PHONY: test docs bench bench-compare lint reproduce numbers

# Run the test suite (CPU-canonical; -n auto comes from pyproject).
test:
	JAX_PLATFORMS=cpu pytest

# Lint and format-check everything CI checks.
lint:
	ruff check src tests scripts benchmarks examples
	ruff format --check src tests scripts benchmarks examples

# Build the MkDocs site with strict mode (any warning fails the build).
docs:
	mkdocs build --strict

# Regenerate all cached analysis outputs and paper figures/tables (K1-K10 +).
reproduce:
	JAX_PLATFORMS=cpu ./reproduce_all.sh --stage core

# Regenerate paper_numbers.tex from results/*.json.
numbers:
	JAX_PLATFORMS=cpu PYTHONPATH=scripts python scripts/emit_paper_numbers.py


# Run the asv perf harness (warmup + single timing per benchmark).
bench:
	asv run --quick --show-stderr

# Report per-commit perf deltas (HEAD~1 vs HEAD).
bench-compare:
	asv compare HEAD~1 HEAD || true
