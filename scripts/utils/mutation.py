"""
This script shows examples of how to mutating contrastive sets from a sed set of 
sequences.

It can generate high and low variants of coding sequences based on five different
optimization criteria:

* ``frequent`` – substitute each codon with the most or least frequently used
  synonymous codon according to a provided codon usage table (akin to CAI-based
  optimisation following the intuition of JCat).
* ``gc`` – substitute each codon with the synonymous codon having the highest
  or lowest GC content.
* ``cpg`` – mutate sequences to increase or decrease CpG dinucleotide density
  using simple heuristics on successive codons.
* ``upa`` – similar to ``cpg`` but targeting UpA dinucleotide density.
* ``cufd`` – select codons to match (high) or diverge from (low) the target
  organism's codon usage frequency distribution, enabling steering towards
  optimal translational efficiency.

Users should configure the ``type`` variable and file paths at the top of
the script. Input sequences are expected in FASTA format and should consist
of coding sequences with lengths divisible by three.

Dependencies:
  * Biopython – for reading and writing FASTA files.
  * a ``mapping`` module providing ``aa_to_codon`` and ``codon_to_aa``
    dictionaries (used in cpg/upa modes).
  * a ``calculator`` module providing ``CpG_density`` and ``UpA_density``
    functions (used in cpg/upa modes).

Example:

    type = "frequent"
    data_path = "input_sequences.fa"
    codon_usage_table_path = "codon_adaptiveness.json"
    output_path = "high_cai_sequences.fa"
    output_path2 = "low_cai_sequences.fa"

    # then run
    python integrated_codon_mutation.py

The script will read the input FASTA, compute the requested mutations, and
write two FASTA files containing the high and low variants.
"""

import json
import random
from typing import Dict, Tuple, List

from Bio import SeqIO
from Bio.Seq import Seq

# -----------------------------------------------------------------------------
# Configuration: edit these variables to match your local environment
# -----------------------------------------------------------------------------

type: str = "gc"  # choose from: 'frequent', 'gc', 'cpg', 'upa', 'cufd'

# Path to the input coding sequences (FASTA format). Each sequence length
# should be divisible by 3.
data_path: str = (
    "your_sequences.fa"
)

# Path to a JSON file mapping codons to (amino_acid, frequency). Only used
# when type is 'frequent' or 'gc'.
codon_usage_table_path: str = (
    "your_codon_usage_table.json"
)

# Output FASTA files for the high and low mutated sequences.
output_path: str = (
    "your_high_output.fa"
)
output_path2: str = (
    "your_low_output.fa"
)

# Set a deterministic seed for reproducibility of random choices
random.seed(42)

# -----------------------------------------------------------------------------
# Codon usage optimisation functions
# -----------------------------------------------------------------------------

def get_most_and_least_codon(
    codon_usage_table: Dict[str, Tuple[str, float]]
) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Tuple[str, float]]]:
    """
    Determine the most and least frequent codon for each amino acid.

    Parameters
    ----------
    codon_usage_table : dict
        Mapping from codon (e.g. "ATG") to a tuple (amino_acid, frequency).

    Returns
    -------
    aa_to_most_frequent_codon : dict
        Maps amino acid to a (codon, frequency) pair representing the most
        frequent synonymous codon.
    aa_to_least_frequent_codon : dict
        Maps amino acid to a (codon, frequency) pair representing the least
        frequent synonymous codon.
    """
    aa_to_most_frequent_codon: Dict[str, Tuple[str, float]] = {}
    aa_to_least_frequent_codon: Dict[str, Tuple[str, float]] = {}
    for codon, (aa, freq) in codon_usage_table.items():
        # update most frequent codon for this amino acid
        if aa not in aa_to_most_frequent_codon or freq > aa_to_most_frequent_codon[aa][1]:
            aa_to_most_frequent_codon[aa] = (codon, freq)
        # update least frequent codon for this amino acid
        if aa not in aa_to_least_frequent_codon or freq < aa_to_least_frequent_codon[aa][1]:
            aa_to_least_frequent_codon[aa] = (codon, freq)
    return aa_to_most_frequent_codon, aa_to_least_frequent_codon


def gc_content(codon: str) -> float:
    """Compute the GC content of a codon as a fraction between 0 and 1."""
    return (codon.count("G") + codon.count("C")) / len(codon)


def get_high_and_low_gc_codons(
    codon_usage_table: Dict[str, Tuple[str, float]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Determine the synonymous codons with highest and lowest GC content for each
    amino acid.

    Parameters
    ----------
    codon_usage_table : dict
        Mapping from codon to a tuple (amino_acid, frequency). The frequency
        values are ignored for GC-based optimisation.

    Returns
    -------
    aa_to_high_gc : dict
        Maps amino acid to a codon with the highest GC content. If multiple
        codons share the maximum GC content, one is chosen randomly.
    aa_to_low_gc : dict
        Maps amino acid to a codon with the lowest GC content. If multiple
        codons share the minimum GC content, one is chosen randomly.
    """
    # group codons by amino acid
    aa_to_codons: Dict[str, List[str]] = {}
    for codon, (aa, _) in codon_usage_table.items():
        aa_to_codons.setdefault(aa, []).append(codon)
    # determine high/low GC codons
    aa_to_high_gc: Dict[str, str] = {}
    aa_to_low_gc: Dict[str, str] = {}
    for aa, codons in aa_to_codons.items():
        codons_sorted = sorted(codons, key=gc_content)
        # pick among ties randomly
        low_gc_codons = [c for c in codons_sorted if gc_content(c) == gc_content(codons_sorted[0])]
        high_gc_codons = [c for c in codons_sorted if gc_content(c) == gc_content(codons_sorted[-1])]
        aa_to_low_gc[aa] = random.choice(low_gc_codons)
        aa_to_high_gc[aa] = random.choice(high_gc_codons)
    return aa_to_high_gc, aa_to_low_gc


# -----------------------------------------------------------------------------
# Dinucleotide-based optimisation functions (CpG/UpA)
# -----------------------------------------------------------------------------

try:
    # Attempt to import mapping dictionaries and calculators. These are only
    # required when using cpg/upa optimisation.
    from mapping import aa_to_codon, codon_to_aa  # type: ignore
    from calculator import CpG_density as cpg_calculation  # type: ignore
    from calculator import UpA_density as upa_calculation  # type: ignore
except ImportError:
    # Define dummy placeholders if the modules are unavailable at import time.
    aa_to_codon = None  # type: ignore
    codon_to_aa = None  # type: ignore
    cpg_calculation = None  # type: ignore
    upa_calculation = None  # type: ignore


def codon_selection_dinucleotide(
    aa: str,
    last_codon: str,
    dinuc: str = "CG",
    mode: str = "high",
) -> str:
    """
    Select a codon for a given amino acid based on the presence or absence
    of a specified dinucleotide (e.g., CpG or UpA).

    This function embodies simple heuristics to enrich or deplete a given
    dinucleotide in the resulting sequence. It is based on logic from the
    original scripts provided by the user, adapted to work generically for
    any dinucleotide.

    Parameters
    ----------
    aa : str
        Single-letter amino acid code for which a codon should be selected.
    last_codon : str
        The codon immediately preceding the one being selected. This is used
        to decide on codons that either start with or avoid starting with
        certain bases, depending on the desired dinucleotide enrichment.
    dinuc : str, optional
        Two-letter nucleotide string that defines the dinucleotide to be
        enriched or depleted (default is "CG" for CpG).
    mode : str, optional
        Either "high" or "low" to indicate whether the dinucleotide density
        should be increased or decreased.

    Returns
    -------
    str
        A selected codon sequence.
    """
    if aa_to_codon is None:
        raise ImportError(
            "mapping.aa_to_codon and mapping.codon_to_aa must be available for cpg/upa modes"
        )
    codon_candidates: List[str] = aa_to_codon[aa]
    # Determine whether the previous codon ends with the first base of the dinucleotide
    prev_ends_with = last_codon and last_codon[2] == dinuc[0]
    # Codons that contain the dinucleotide of interest
    contains_dinuc = [c for c in codon_candidates if dinuc in c]
    other_candidates = list(set(codon_candidates) - set(contains_dinuc))
    # High mode: enrich dinucleotide
    if mode == "high":
        # Heuristic special cases from the original scripts
        if dinuc == "CG" and prev_ends_with and aa == "A":
            return "GCG"
        if dinuc == "UA" and prev_ends_with and aa == "I":
            return "AUA"
        # If previous codon ends with the first base but there are no codons containing dinuc,
        # pick a codon starting with the second base of the dinuc
        if prev_ends_with and not contains_dinuc:
            candidates = [c for c in codon_candidates if c.startswith(dinuc[1])]
            return random.choice(candidates) if candidates else random.choice(codon_candidates)
        # If previous codon does not end with first base and there are codons containing dinuc,
        # pick one containing the dinuc
        if not prev_ends_with and contains_dinuc:
            return random.choice(contains_dinuc)
        # Otherwise, try to pick a codon ending with the first base of the dinuc
        if not prev_ends_with and not contains_dinuc:
            candidates = [c for c in codon_candidates if c.endswith(dinuc[0])]
            return random.choice(candidates) if candidates else random.choice(codon_candidates)
    # Low mode: deplete dinucleotide
    elif mode == "low":
        # Filter out any codon that ends with the first base of the dinucleotide
        candidates = [c for c in other_candidates if not c.endswith(dinuc[0])]
        # If previous codon ends with the first base, also avoid codons starting with the second base
        if prev_ends_with:
            candidates = [c for c in candidates if not c.startswith(dinuc[1])]
        if candidates:
            return random.choice(candidates)
    # Fallback: return a random codon
    return random.choice(codon_candidates)


def get_new_sequence_dinucleotide(
    seq: str,
    dinuc: str = "CG",
    mode: str = "high",
) -> str:
    """
    Build a new sequence with adjusted dinucleotide (CpG or UpA) density.

    Parameters
    ----------
    seq : str
        Original nucleotide sequence. Thymine bases will be converted to
        uracil for codon lookup.
    dinuc : str, optional
        Dinucleotide to enrich or deplete ("CG" for CpG or "UA" for UpA).
    mode : str, optional
        Either "high" or "low" to control enrichment or depletion.

    Returns
    -------
    str
        The mutated sequence as a string of codons.
    """
    new_seq = ""
    last_codon = ""
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3].replace("T", "U")
        if codon not in codon_to_aa:
            raise ValueError(f"Invalid codon {codon} in sequence")
        aa = codon_to_aa[codon]
        new_codon = codon_selection_dinucleotide(aa, last_codon, dinuc=dinuc, mode=mode)
        new_seq += new_codon
        last_codon = new_codon
    return new_seq


# -----------------------------------------------------------------------------
# CUFD-based optimisation functions
# -----------------------------------------------------------------------------

def cufd_codon_selection(
    aa: str,
    codon_usage_table: dict,
    mode: str = "high",
) -> str:
    """
    Select a codon for a given amino acid based on the organism's codon usage
    frequency distribution (CUFD).

    Parameters
    ----------
    aa : str
        Three-letter amino acid code (e.g., "Phe").
    codon_usage_table : dict
        Mapping from codon to {"aa": ..., "freq_per_1000": ...}.
    mode : str, optional
        "high" to sample proportional to organism frequencies (good match),
        "low" to always pick the rarest codon (poor match).

    Returns
    -------
    str
        The selected codon.
    """
    # Collect codons for this amino acid
    codons_for_aa = [
        (codon, entry["freq_per_1000"])
        for codon, entry in codon_usage_table.items()
        if entry["aa"] == aa
    ]
    if not codons_for_aa:
        raise ValueError(f"No codons found for amino acid {aa}")

    if mode == "high":
        # Weighted random selection proportional to organism frequency
        codons, weights = zip(*codons_for_aa)
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(codons, weights=probs, k=1)[0]
    elif mode == "low":
        # Always pick the rarest codon
        codons_for_aa.sort(key=lambda x: x[1])
        return codons_for_aa[0][0]
    else:
        raise ValueError(f"mode must be 'high' or 'low', got '{mode}'")


def get_new_sequence_cufd(
    seq: str,
    codon_usage_table: dict,
    codon_to_amino_acid: dict,
    mode: str = "high",
) -> str:
    """
    Build a new sequence with codons selected to match (high) or diverge from
    (low) the target organism's codon usage frequency distribution.

    Parameters
    ----------
    seq : str
        Original nucleotide sequence (length divisible by 3).
    codon_usage_table : dict
        Organism codon usage table with freq_per_1000 values.
    codon_to_amino_acid : dict
        Mapping from codon to three-letter amino acid code.
    mode : str, optional
        "high" for organism-matched, "low" for organism-divergent.

    Returns
    -------
    str
        The mutated sequence.
    """
    new_seq = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3].upper().replace("T", "U")
        if codon not in codon_to_amino_acid:
            new_seq += codon
            continue
        aa = codon_to_amino_acid[codon]
        new_codon = cufd_codon_selection(aa, codon_usage_table, mode=mode)
        new_seq += new_codon
    return new_seq


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Entry point for the script. Performs the requested optimisation and writes output files."""
    # Read input sequences
    records = list(SeqIO.parse(data_path, "fasta"))
    # Prepare containers for high and low variants
    high_records: List = []
    low_records: List = []

    # When optimising codon usage ('frequent' or 'gc'), load codon usage table
    if type in ("frequent", "gc"):
        with open(codon_usage_table_path) as f:
            codon_usage_table = json.load(f)
        # Map each codon to its amino acid for fast lookup
        codon_to_amino_acid = {codon: aa for codon, (aa, _) in codon_usage_table.items()}

    # Handle each optimisation type separately
    if type == "frequent":
        aa_to_most, aa_to_least = get_most_and_least_codon(codon_usage_table)
        for record in records:
            seq = record.seq
            # Build high-CAI sequence
            high_seq = ""
            for i in range(0, len(seq), 3):
                codon = str(seq[i : i + 3])
                if codon in codon_to_amino_acid:
                    aa = codon_to_amino_acid[codon]
                    high_seq += aa_to_most.get(aa, (codon,))[0]
                else:
                    high_seq += codon
            high_rec = record.copy()
            high_rec.seq = Seq(high_seq)
            high_rec.description = (high_rec.description or "") + " high_cai"
            high_records.append(high_rec)
            # Build low-CAI sequence
            low_seq = ""
            for i in range(0, len(seq), 3):
                codon = str(seq[i : i + 3])
                if codon in codon_to_amino_acid:
                    aa = codon_to_amino_acid[codon]
                    low_seq += aa_to_least.get(aa, (codon,))[0]
                else:
                    low_seq += codon
            low_rec = record.copy()
            low_rec.seq = Seq(low_seq)
            low_rec.description = (low_rec.description or "") + " low_cai"
            low_records.append(low_rec)

    elif type == "gc":
        aa_to_high_gc, aa_to_low_gc = get_high_and_low_gc_codons(codon_usage_table)
        for record in records:
            seq = record.seq
            high_seq = ""
            low_seq = ""
            for i in range(0, len(seq), 3):
                codon = str(seq[i : i + 3])
                if codon in codon_to_amino_acid:
                    aa = codon_to_amino_acid[codon]
                    high_seq += aa_to_high_gc.get(aa, codon)
                    low_seq += aa_to_low_gc.get(aa, codon)
                else:
                    high_seq += codon
                    low_seq += codon
            high_rec = record.copy()
            high_rec.seq = Seq(high_seq)
            high_rec.description = (high_rec.description or "") + " high_gc"
            high_records.append(high_rec)
            low_rec = record.copy()
            low_rec.seq = Seq(low_seq)
            low_rec.description = (low_rec.description or "") + " low_gc"
            low_records.append(low_rec)

    elif type in ("cpg", "upa"):
        # Ensure required modules were imported
        if aa_to_codon is None or codon_to_aa is None:
            raise ImportError(
                "The mapping module with aa_to_codon and codon_to_aa must be available for cpg/upa modes."
            )
        if type == "cpg":
            dinuc = "CG"
            calc_function = cpg_calculation
        else:
            dinuc = "UA"
            calc_function = upa_calculation
        # Compute high and low variants
        for record in records:
            seq_str = str(record.seq)
            high_seq = get_new_sequence_dinucleotide(seq_str, dinuc=dinuc, mode="high")
            low_seq = get_new_sequence_dinucleotide(seq_str, dinuc=dinuc, mode="low")
            high_rec = record.copy()
            high_rec.seq = Seq(high_seq)
            high_rec.description = (high_rec.description or "") + f" high_{type}"
            high_records.append(high_rec)
            low_rec = record.copy()
            low_rec.seq = Seq(low_seq)
            low_rec.description = (low_rec.description or "") + f" low_{type}"
            low_records.append(low_rec)
        # Optionally report mean dinucleotide densities
        if calc_function is not None:
            original_scores = [calc_function(str(rec.seq)) for rec in records]
            high_scores = [calc_function(str(rec.seq)) for rec in high_records]
            low_scores = [calc_function(str(rec.seq)) for rec in low_records]
            print(
                f"Mean {type} score of original sequences: {sum(original_scores) / len(original_scores):.4f}"
            )
            print(
                f"Mean {type} score of high mutated sequences: {sum(high_scores) / len(high_scores):.4f}"
            )
            print(
                f"Mean {type} score of low mutated sequences: {sum(low_scores) / len(low_scores):.4f}"
            )
    elif type == "cufd":
        # Load organism codon usage table (uses the same format as codon_usage_human.json)
        import os
        cufd_table_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "calculator", "data", "codon_usage_human.json"
        )
        with open(cufd_table_path) as f:
            cufd_table = json.load(f)
        # Build codon→aa mapping from the CUFD table
        cufd_codon_to_aa = {codon: entry["aa"] for codon, entry in cufd_table.items()}
        for record in records:
            seq_str = str(record.seq)
            high_seq = get_new_sequence_cufd(seq_str, cufd_table, cufd_codon_to_aa, mode="high")
            low_seq = get_new_sequence_cufd(seq_str, cufd_table, cufd_codon_to_aa, mode="low")
            high_rec = record.copy()
            high_rec.seq = Seq(high_seq)
            high_rec.description = (high_rec.description or "") + " high_cufd"
            high_records.append(high_rec)
            low_rec = record.copy()
            low_rec.seq = Seq(low_seq)
            low_rec.description = (low_rec.description or "") + " low_cufd"
            low_records.append(low_rec)

    else:
        raise ValueError(
            "Unknown type: {}. Expected one of 'frequent', 'gc', 'cpg', 'upa', 'cufd'.".format(type)
        )

    # Write out the mutated sequences
    SeqIO.write(high_records, output_path, "fasta")
    SeqIO.write(low_records, output_path2, "fasta")
    print(
        f"Saved {len(high_records)} high mutated sequences to {output_path}\n"
        f"Saved {len(low_records)} low mutated sequences to {output_path2}"
    )


if __name__ == "__main__":
    main()