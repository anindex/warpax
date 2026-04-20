.PHONY: test docs bench bench-compare

# Run the full test suite (CPU-canonical; skip slow marker).
test:
	JAX_PLATFORMS=cpu pytest -m "not slow"

# Build the MkDocs site with strict mode (any warning fails the build).
docs:
	mkdocs build --strict

# Run the asv perf harness (warmup + single timing per benchmark).
bench:
	asv run --quick --show-stderr

# Report per-commit perf deltas (HEAD~1 vs HEAD).
bench-compare:
	asv compare HEAD~1 HEAD || true
