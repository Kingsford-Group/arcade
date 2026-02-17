"""
Tests for the ARCADE calculator module.

Covers: gc_content, CpG_density, UpA_density, cai_calculation,
        cufd_kl_divergence, cufd_cosine_similarity, and edge cases.
"""
import os
import sys
import math
import types
import pytest

# Stub out ViennaRNA (RNA) so the calculator module can be imported
# without the native C library being installed.
if "RNA" not in sys.modules:
    _rna_stub = types.ModuleType("RNA")
    _rna_stub.fold = lambda seq: ("." * len(seq), 0.0)  # type: ignore[attr-defined]
    sys.modules["RNA"] = _rna_stub

# Make calculator importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calculator"))
from calculator import (
    gc_content,
    CpG_density,
    UpA_density,
    cai_calculation,
    cufd_kl_divergence,
    cufd_cosine_similarity,
)


# ===================================================================
# Tests for gc_content
# ===================================================================

class TestGCContent:
    """Tests for the gc_content function."""

    def test_all_gc(self):
        """A sequence of only G and C should have GC content = 1.0."""
        assert gc_content("GCGCGCGC") == 1.0

    def test_all_at(self):
        """A sequence of only A and T should have GC content = 0.0."""
        assert gc_content("ATATATATAT") == 0.0

    def test_half_gc(self):
        """A balanced sequence should have GC content ≈ 0.5."""
        assert gc_content("ATGC") == pytest.approx(0.5)

    def test_empty_sequence(self):
        """An empty sequence should return 0."""
        assert gc_content("") == 0

    def test_known_value(self):
        """Known short sequence with hand-computed GC content."""
        # ATGGCC → G:2 + C:2 = 4/6 ≈ 0.6667
        assert gc_content("ATGGCC") == pytest.approx(4.0 / 6.0)

    def test_rna_input(self):
        """GC content should work with U bases (RNA)."""
        # AUGCCC → 0G→wait, 1G? No: A-U-G-C-C-C = G:1, C:3 → 4/6
        assert gc_content("AUGCCC") == pytest.approx(4.0 / 6.0)


# ===================================================================
# Tests for CpG_density
# ===================================================================

class TestCpGDensity:
    """Tests for the CpG_density function."""

    def test_no_cpg(self):
        """A sequence with no CG dinucleotides."""
        assert CpG_density("AAAAATTTTT") == 0.0

    def test_known_cpg(self):
        """Hand-counted CpG density."""
        # "CGCGCG" → CG appears at positions 0,2,4 → 3 occurrences
        # 3/6 * 100 = 50.0
        assert CpG_density("CGCGCG") == pytest.approx(50.0)

    def test_single_cpg(self):
        """One CG in a longer sequence."""
        # "AACGTT" → 1 CG, len=6 → 1/6 * 100 ≈ 16.67
        assert CpG_density("AACGTT") == pytest.approx(100.0 / 6.0)

    def test_empty_sequence(self):
        """Empty sequence returns 0."""
        assert CpG_density("") == 0


# ===================================================================
# Tests for UpA_density
# ===================================================================

class TestUpADensity:
    """Tests for the UpA_density function."""

    def test_no_upa(self):
        """A sequence with no UA/TA dinucleotides."""
        assert UpA_density("GCGCGCGC") == 0.0

    def test_known_upa(self):
        """Hand-counted UpA density with T→U conversion."""
        # "TATATA" → T→U → "UAUAUA" → UA at pos 0,2,4 → 3 occurrences
        # 3/6 * 100 = 50.0
        assert UpA_density("TATATA") == pytest.approx(50.0)

    def test_rna_input(self):
        """Input already in RNA (with U)."""
        # "UAUAUA" → UA at pos 0,2,4 → 3/6*100 = 50
        assert UpA_density("UAUAUA") == pytest.approx(50.0)

    def test_empty_sequence(self):
        """Empty sequence returns 0."""
        assert UpA_density("") == 0

    def test_single_upa(self):
        """One TA in a longer sequence."""
        # "GGTACC" → T→U → "GGUACC" → UA at pos 2 → 1/6*100 ≈ 16.67
        assert UpA_density("GGTACC") == pytest.approx(100.0 / 6.0)


# ===================================================================
# Tests for cai_calculation
# ===================================================================

class TestCAICalculation:
    """Tests for the cai_calculation function."""

    def test_single_met_codon(self, codon_adaptiveness_path, max_codon_adaptiveness_path):
        """ATG (Met) has fraction 1.0 and max 1.0, so CAI = 1.0."""
        cai = cai_calculation("ATG", codon_adaptiveness_path, max_codon_adaptiveness_path)
        assert cai == pytest.approx(1.0)

    def test_optimal_codons_high_cai(self, codon_adaptiveness_path, max_codon_adaptiveness_path):
        """Using the most frequent codon per AA should yield high CAI."""
        # ATG (Met, 1.0) + GCC (Ala, most freq 0.398) + CTG (Leu, most freq 0.391)
        seq = "ATGGCCCTG"
        cai = cai_calculation(seq, codon_adaptiveness_path, max_codon_adaptiveness_path)
        assert cai > 0.8

    def test_suboptimal_codons_lower_cai(self, codon_adaptiveness_path, max_codon_adaptiveness_path):
        """Using rare codons should yield lower CAI."""
        # ATG (Met) + GCG (Ala, rare) + CTA (Leu, rare)
        seq = "ATGGCGCTA"
        cai = cai_calculation(seq, codon_adaptiveness_path, max_codon_adaptiveness_path)
        assert cai < 0.7


# ===================================================================
# Tests for cufd_kl_divergence
# ===================================================================

class TestCUFDKLDivergence:
    """Tests for the cufd_kl_divergence function."""

    def test_empty_sequence(self, codon_usage_human_path):
        """Empty sequence returns 0.0."""
        assert cufd_kl_divergence("", codon_usage_human_path) == 0.0

    def test_single_codon(self, codon_usage_human_path):
        """Single codon: only one amino acid, distribution is a delta."""
        # ATG = Met (only one codon → always matches perfectly)
        kl = cufd_kl_divergence("ATG", codon_usage_human_path)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self, codon_usage_human_path):
        """KL divergence must be non-negative."""
        kl = cufd_kl_divergence("ATGGCCGCAGCGGCT", codon_usage_human_path)
        assert kl >= 0.0

    def test_multiple_amino_acids(self, codon_usage_human_path):
        """CUFD works with multiple amino acids in the sequence."""
        seq = "ATGGCCGGCCTGTTCTCC"  # Met-Ala-Gly-Leu-Phe-Ser
        kl = cufd_kl_divergence(seq, codon_usage_human_path)
        assert isinstance(kl, float)
        assert kl >= 0.0

    def test_stop_codons_excluded(self, codon_usage_human_path):
        """Stop codons should not contribute to the CUFD score."""
        # Met + Stop
        kl = cufd_kl_divergence("ATGTAA", codon_usage_human_path)
        # Only Met contributes; Met has 1 codon → KL ≈ 0
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_repeated_optimal_low_kl(self, codon_usage_human_path):
        """Repeating the most frequent codon for an AA with multiple synonymous
        codons should produce a non-zero (but potentially low) KL."""
        # All GCC (Ala most frequent in humans)
        seq = "GCCGCCGCCGCCGCCGCCGCCGCCGCCGCC"
        kl = cufd_kl_divergence(seq, codon_usage_human_path)
        assert kl >= 0.0


# ===================================================================
# Tests for cufd_cosine_similarity
# ===================================================================

class TestCUFDCosineSimilarity:
    """Tests for the cufd_cosine_similarity function."""

    def test_empty_sequence(self, codon_usage_human_path):
        """Empty sequence returns 0.0."""
        assert cufd_cosine_similarity("", codon_usage_human_path) == 0.0

    def test_single_codon_perfect(self, codon_usage_human_path):
        """Single codon for a 1-codon AA → cosine similarity = 1.0."""
        cos_sim = cufd_cosine_similarity("ATG", codon_usage_human_path)
        assert cos_sim == pytest.approx(1.0, abs=1e-6)

    def test_range_zero_to_one(self, codon_usage_human_path):
        """Cosine similarity should be between 0 and 1."""
        seq = "ATGGCCGGCCTGTTCTCC"
        cos_sim = cufd_cosine_similarity(seq, codon_usage_human_path)
        assert 0.0 <= cos_sim <= 1.0

    def test_all_same_codon_multicodon_aa(self, codon_usage_human_path):
        """Using only one synonym for a multi-codon AA gives a valid score."""
        # All GCC (Ala, 4 synonymous codons)
        seq = "GCCGCCGCCGCCGCC"
        cos_sim = cufd_cosine_similarity(seq, codon_usage_human_path)
        assert 0.0 < cos_sim <= 1.0

    def test_cosine_vs_kl_consistency(self, codon_usage_human_path):
        """When KL divergence is near-zero, cosine similarity should be near 1.0."""
        # Met only has 1 codon → both should be trivial
        kl = cufd_kl_divergence("ATGATGATG", codon_usage_human_path)
        cos_sim = cufd_cosine_similarity("ATGATGATG", codon_usage_human_path)
        assert kl == pytest.approx(0.0, abs=1e-6)
        assert cos_sim == pytest.approx(1.0, abs=1e-6)

    def test_rna_input(self, codon_usage_human_path):
        """Should handle RNA input (U instead of T)."""
        cos_sim = cufd_cosine_similarity("AUGGCCGGCCUG", codon_usage_human_path)
        assert 0.0 <= cos_sim <= 1.0
