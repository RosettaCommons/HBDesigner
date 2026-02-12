import pytest 
from pathlib import Path
import os 
import numpy as np
from numpy.testing import assert_allclose

from hbdesigner.data.protein import Protein, PDB_CHAIN_IDS
from hbdesigner.data.hbnet import initialize_rosetta, get_guide_atom
from hbdesigner.inference.protein_ops import (
    extract_chains,
    get_core_mask,
    get_symmetry_mask,
    validate_residues,
    validate_chains,
    add_guide_atom,
    concat_proteins,
    symmetrize_output,
    get_network_res,
)

from test_data_utils import example_protein


initialize_rosetta()
np.random.seed(42)


@pytest.fixture
def example_dimer() -> Protein:
    """Enables multiple protein instances to be generated."""

    class ProteinFactory(object):
        def get(self, discard_Hs=False) -> Protein:
            pdb_file = os.path.join(
                Path(__file__).parents[1], "examples/interface/1YRK.pdb"
            )
            return Protein.from_pdb_file(pdb_file, discard_Hs=discard_Hs)

    return ProteinFactory()


@pytest.mark.parametrize("sel_chains,used_chains", 
                         [("A", [0]), ("A,B", [0, 1]), ("B,A", [0, 1]), ("B", [1]), ("", [])], 
                         )
def test_extract_chains(example_dimer, sel_chains, used_chains):
    # These are ops used for inference
    all_chains = set([0, 1])
    unused_chains = list(all_chains - set(used_chains))
    protein = example_dimer.get()

    # Chain extraction should return used and unused chains
    p_used, p_unused = extract_chains(protein, sel_chains)
    assert np.unique(p_used.chain_index).tolist() == used_chains
    assert np.unique(p_unused.chain_index).tolist() == unused_chains


def test_concat_proteins(example_dimer):
    protein = example_dimer.get()
    protein_A, protein_B = extract_chains(protein, "A")

    # Sort doesn't change anything if already in order
    p_concat_unsorted = concat_proteins([protein_A, protein_B], sort=False)
    p_concat_sorted = concat_proteins([protein_A, protein_B], sort=True)
    assert p_concat_sorted.n_res == protein.n_res == p_concat_unsorted.n_res
    assert np.all(p_concat_sorted.chain_index == p_concat_unsorted.chain_index)

    # Sort should fix things if out of order
    p_concat_unsorted = concat_proteins([protein_B, protein_A], sort=False)
    p_concat_sorted = concat_proteins([protein_B, protein_A], sort=True)
    assert np.all(p_concat_sorted.chain_index == protein.chain_index)
    assert not np.all(p_concat_unsorted.chain_index == protein.chain_index)

    # Concat should preserve chain IDs and allow recovery of original proteins
    p_recover_B, p_recover_A = extract_chains(p_concat_sorted, "B")
    assert p_recover_B.n_res == protein_B.n_res
    assert p_recover_A.n_res == protein_A.n_res
    assert np.unique(p_recover_B.chain_index).tolist() == [1]
    assert np.unique(p_recover_A.chain_index).tolist() == [0]

    # Duplicate chain indices should throw an error
    with pytest.raises(ValueError):
        concat_proteins([protein_A, protein_A], sort=False)
    with pytest.raises(ValueError):
        concat_proteins([protein_A, protein], sort=True)


def test_get_core_mask(example_protein):
    # These are ops used for inference
    protein = example_protein.get()

    # Core mask should be boolean and have same length as number of residues
    core_mask = get_core_mask(protein)
    assert isinstance(core_mask, np.ndarray)
    assert core_mask.dtype == bool
    assert len(core_mask) == protein.n_res

    # Core mask should be deterministic
    core_mask_2 = get_core_mask(protein)
    assert np.array_equal(core_mask, core_mask_2)


def test_validate_chain(example_dimer):
    protein = example_dimer.get()
    # Validate_chains returns a mask of designable positions
    # This mask is False for any provided chain(s)
    protein.clear_sequence()
    protein.aatype[:] = 20
    mask = validate_chains(protein, "A")
    assert np.all(protein.chain_index[mask] == [1])
    mask = validate_chains(protein, "B")
    assert np.all(protein.chain_index[mask] == [0])
    mask = validate_chains(protein, "A,B")
    assert np.unique(protein.chain_index[mask]).tolist() == []
    mask = validate_chains(protein)
    assert np.unique(protein.chain_index[mask]).tolist() == [0, 1]

    # Invalid chain should raise an error
    with pytest.raises(ValueError):
        validate_chains(protein, "C")
    with pytest.raises(ValueError):
        validate_chains(protein, "A,B,C")


def test_validate_res(example_dimer):
    protein = example_dimer.get()
    # Guide mode just checks if residues exist in Protein
    guide_res = "A1,A2,B3"
    res_array = validate_residues(protein, "A1,A2,B3", mode="guide")
    assert len(res_array) == len(guide_res.split(","))
    # Should throw an error if they don't, or if there are less than 2 res provided
    with pytest.raises(ValueError):
        validate_residues(protein, "A1", mode="guide")
    with pytest.raises(ValueError):
        validate_residues(protein, "A1000", mode="guide")
    with pytest.raises(ValueError):
        validate_residues(protein, "C2", mode="guide")

    # Fixed mode checks if residues have polar aatypes too
    fixed_res = "B5"
    res_array = validate_residues(protein, fixed_res, mode="fixed")
    assert len(res_array) == len(fixed_res.split(","))
    # Should throw an error if nonpolar residues are included
    fixed_res = "A1,B3,B5"
    with pytest.raises(ValueError):
        validate_residues(protein, fixed_res, mode="fixed")


def test_guide_atom(example_protein):
    protein = example_protein.get()

    # Adding guide atom stores it in the hetatm_dict attr of Protein
    centroid = np.array([10.0, 10.0, 10.0])
    protein = add_guide_atom(protein, centroid)
    assert len(protein.hetatm_dict.keys()) > 0
    assert "V1" in protein.hetatm_dict["atom_name"]
    assert "V" in protein.hetatm_dict["element"]
    assert "ORI" in protein.hetatm_dict["res_name"]
    assert_allclose(protein.hetatm_dict["atom_xyz"], centroid)

    # Guide atom can be retrieved from backbone coords of 3+ residues
    guide_atom_res = np.zeros((6, 3, 3)) #[N, 3, 3]
    guide_atom_xyz = get_guide_atom(guide_atom_res, sigma=0.0)
    assert guide_atom_xyz.shape == (1, 3,)
    # Larger sigma means more noise, so should not be exactly the same as sigma=0.0 case, or a second sample
    guide_atom_xyz_noisy = get_guide_atom(guide_atom_res, sigma=10.0)
    assert np.all(guide_atom_xyz != guide_atom_xyz_noisy)
    guide_atom_xyz_noisy_2 = get_guide_atom(guide_atom_res, sigma=10.0)
    assert np.all(guide_atom_xyz_noisy != guide_atom_xyz_noisy_2)

# TODO symmetrize protein
# TODO symmetry mask
# TODO seq_cond_inf (different settings)
# TODO run minimize
# TODO run scoring

