"""Repository locations shared by every script, as absolute path strings."""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
ARXIV_DIR = os.path.join(os.path.dirname(REPO_ROOT), "warpax_arxiv")
TABLES_DIR = os.path.join(ARXIV_DIR, "tables")
