"""
Tests for the codon ↔ amino acid mapping tables in scripts/utils/mapping.py.
"""
import os
import sys
import pytest

# Make mapping importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "utils"))
from mapping import aa_to_codon, codon_to_aa


class TestAminoAcidToCodon:
    """Tests for the aa_to_codon mapping."""

    def test_all_20_amino_acids_plus_stop(self):
        """The table should have entries for 20 amino acids + STOP."""
        assert len(aa_to_codon) == 21

    def test_total_codons_is_64(self):
        """All synonymous codons should add up to 64."""
        total = sum(len(codons) for codons in aa_to_codon.values())
        assert total == 64

    def test_met_single_codon(self):
        """Methionine should have exactly one codon: AUG."""
        assert aa_to_codon["M"] == ["AUG"]

    def test_trp_single_codon(self):
        """Tryptophan should have exactly one codon: UGG."""
        assert aa_to_codon["W"] == ["UGG"]

    def test_stop_codons(self):
        """Stop codons should be UAA, UAG, UGA."""
        assert set(aa_to_codon["STOP"]) == {"UAA", "UAG", "UGA"}

    def test_leucine_six_codons(self):
        """Leucine should be encoded by 6 codons."""
        assert len(aa_to_codon["L"]) == 6

    def test_serine_six_codons(self):
        """Serine should be encoded by 6 codons."""
        assert len(aa_to_codon["S"]) == 6

    def test_arginine_six_codons(self):
        """Arginine should be encoded by 6 codons."""
        assert len(aa_to_codon["R"]) == 6

    def test_all_codons_are_triplets(self):
        """Every codon in the table should be exactly 3 characters."""
        for aa, codons in aa_to_codon.items():
            for codon in codons:
                assert len(codon) == 3, f"Codon {codon} for {aa} is not a triplet"

    def test_codons_use_rna_alphabet(self):
        """All codons should only contain A, U, G, C (RNA)."""
        valid = set("AUGC")
        for aa, codons in aa_to_codon.items():
            for codon in codons:
                assert set(codon).issubset(valid), f"Codon {codon} has invalid chars"


class TestCodonToAminoAcid:
    """Tests for the codon_to_aa reverse mapping."""

    def test_total_entries_is_64(self):
        """The reverse mapping should have exactly 64 entries (one per codon)."""
        assert len(codon_to_aa) == 64

    def test_reverse_mapping_consistency(self):
        """Every codon→aa mapping should be consistent with aa→codon."""
        for codon, aa in codon_to_aa.items():
            assert codon in aa_to_codon[aa], (
                f"Codon {codon} maps to {aa}, but is not in aa_to_codon[{aa}]"
            )

    def test_forward_reverse_roundtrip(self):
        """Every codon listed in aa_to_codon should map back to its AA
        in codon_to_aa."""
        for aa, codons in aa_to_codon.items():
            for codon in codons:
                assert codon_to_aa[codon] == aa

    def test_aug_maps_to_met(self):
        """AUG should map to M (Methionine)."""
        assert codon_to_aa["AUG"] == "M"

    def test_stop_codon_mapping(self):
        """All three stop codons should map to 'STOP'."""
        for codon in ["UAA", "UAG", "UGA"]:
            assert codon_to_aa[codon] == "STOP"
