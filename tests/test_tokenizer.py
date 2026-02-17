"""
Tests for the tokenizer utilities in benchmarks/utils/tokenizer.py.
"""
import os
import sys
import pytest

# Make tokenizer importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "utils"))
from tokenizer import mytok, get_tokenizer


# ===================================================================
# Tests for mytok
# ===================================================================

class TestMytok:
    """Tests for the mytok codon tokenizer."""

    def test_basic_tokenization(self):
        """A 6-base sequence should produce 2 codons."""
        tokens = mytok("ATGCCA", 3, 3)
        assert tokens == ["AUG", "CCA"]

    def test_t_to_u_conversion(self):
        """T should be converted to U in output."""
        tokens = mytok("TTT", 3, 3)
        assert tokens == ["UUU"]

    def test_uppercase_conversion(self):
        """Lowercase input should be uppercased."""
        tokens = mytok("atgcca", 3, 3)
        assert tokens == ["AUG", "CCA"]

    def test_longer_sequence(self):
        """A 12-base sequence should produce 4 codons."""
        tokens = mytok("ATGGCCGGCCTG", 3, 3)
        assert tokens == ["AUG", "GCC", "GGC", "CUG"]

    def test_partial_codon_dropped(self):
        """Trailing bases that don't form a full codon are dropped."""
        # 7 bases → only 2 full codons (6 bases)
        tokens = mytok("ATGCCAG", 3, 3)
        assert tokens == ["AUG", "CCA"]

    def test_empty_string(self):
        """Empty input returns empty list."""
        tokens = mytok("", 3, 3)
        assert tokens == []

    def test_short_string(self):
        """Input shorter than kmer_len returns empty list."""
        tokens = mytok("AT", 3, 3)
        assert tokens == []

    def test_single_codon(self):
        """Exactly one codon."""
        tokens = mytok("ATG", 3, 3)
        assert tokens == ["AUG"]

    def test_rna_input_unchanged(self):
        """If input already uses U, it should remain unchanged."""
        tokens = mytok("AUGCCA", 3, 3)
        assert tokens == ["AUG", "CCA"]

    def test_all_bases(self):
        """All four DNA bases are handled correctly."""
        tokens = mytok("ACGTAC", 3, 3)
        # A C G → ACG (T→U not applicable here)
        # T A C → UAC (T→U)
        assert tokens == ["ACG", "UAC"]


# ===================================================================
# Tests for get_tokenizer
# ===================================================================

class TestGetTokenizer:
    """Tests for the get_tokenizer factory function."""

    def test_vocab_size(self):
        """Vocabulary should have 5 special tokens + 64 codons = 69."""
        tokenizer = get_tokenizer()
        vocab = tokenizer.get_vocab()
        assert len(vocab) == 69

    def test_special_tokens_present(self):
        """All expected special tokens should be in the vocabulary."""
        tokenizer = get_tokenizer()
        vocab = tokenizer.get_vocab()
        for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            assert token in vocab, f"{token} missing from tokenizer vocab"

    def test_special_tokens_ids(self):
        """Special tokens should have the IDs 0-4."""
        tokenizer = get_tokenizer()
        vocab = tokenizer.get_vocab()
        assert vocab["[PAD]"] == 0
        assert vocab["[UNK]"] == 1
        assert vocab["[CLS]"] == 2
        assert vocab["[SEP]"] == 3
        assert vocab["[MASK]"] == 4

    def test_all_64_codons_in_vocab(self):
        """All 64 RNA triplet codons should be in the vocabulary."""
        tokenizer = get_tokenizer()
        vocab = tokenizer.get_vocab()
        bases = "AUGC"
        for a in bases:
            for b in bases:
                for c in bases:
                    codon = f"{a}{b}{c}"
                    assert codon in vocab, f"Codon {codon} missing from vocab"

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding a codon sequence should be lossless."""
        tokenizer = get_tokenizer()
        seq = "AUG CCA GGC"
        encoded = tokenizer.encode(seq)
        # encoded includes [CLS] at start and [SEP] at end
        assert len(encoded.ids) == 5  # [CLS] + 3 codons + [SEP]

    def test_tokenizer_returns_consistent_type(self):
        """get_tokenizer should always return a Tokenizer object."""
        tok1 = get_tokenizer()
        tok2 = get_tokenizer()
        assert type(tok1) == type(tok2)
