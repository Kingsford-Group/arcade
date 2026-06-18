"""
Shared test fixtures for the ARCADE test suite.
"""
import os
import sys
import json
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALCULATOR_DIR = os.path.join(REPO_ROOT, "calculator")
CALCULATOR_DATA_DIR = os.path.join(CALCULATOR_DIR, "data")

# Ensure calculator module is importable
sys.path.insert(0, CALCULATOR_DIR)

# Ensure benchmarks/utils (for tokenizer) is importable
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks", "utils"))

# Ensure scripts/utils is importable
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "utils"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def codon_adaptiveness_path():
    """Path to the human codon adaptiveness JSON file."""
    return os.path.join(CALCULATOR_DATA_DIR, "codon_adaptiveness.json")


@pytest.fixture
def max_codon_adaptiveness_path():
    """Path to the max codon adaptiveness JSON file."""
    return os.path.join(CALCULATOR_DATA_DIR, "max_codon_adaptiveness.json")


@pytest.fixture
def codon_usage_human_path():
    """Path to the human codon usage frequency JSON file."""
    return os.path.join(CALCULATOR_DATA_DIR, "codon_usage_human.json")


@pytest.fixture
def codon_adaptiveness_table(codon_adaptiveness_path):
    """Loaded codon adaptiveness table as a dict."""
    with open(codon_adaptiveness_path, "r") as f:
        return json.load(f)


@pytest.fixture
def codon_usage_human_table(codon_usage_human_path):
    """Loaded human codon usage table as a dict."""
    with open(codon_usage_human_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sample sequences for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_seq_atg():
    """Single start codon (Met)."""
    return "ATGCCA"


@pytest.fixture
def sample_seq_gc_rich():
    """A GC-rich sequence (all GCC = Ala)."""
    return "GCCGCCGCCGCCGCCGCC"


@pytest.fixture
def sample_seq_at_rich():
    """An AT-rich sequence (all AAA = Lys)."""
    return "AAAAAAAAAAAAAAAAAA"


@pytest.fixture
def sample_seq_mixed():
    """A mixed coding sequence with several amino acids."""
    # Met-Ala-Gly-Leu-Phe-Ser (ATG GCC GGC CTG TTC TCC)
    return "ATGGCCGGCCTGTTCTCC"
