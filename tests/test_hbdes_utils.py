import pytest 
from pathlib import Path
import os 
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from hbdesigner.data.protein import Protein
from hbdesigner.data import residue_constants as rc
from hbdesigner.data.hbnet import (
    initialize_rosetta, 
    get_guide_atom, get_seq_cond_inf
)
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
    get_symmetry_idx,
)
from test_data_utils import example_protein


initialize_rosetta()
np.random.seed(42)


@pytest.mark.parametrize("sel_chains,used_chains", 
                         [("A", [0]), ("A,B", [0, 1]), ("B,A", [0, 1]), ("B", [1]), ("", [])], 
                         )
def test_extract_chains(example_protein, sel_chains, used_chains):
    # These are ops used for inference
    all_chains = set([0, 1])
    unused_chains = list(all_chains - set(used_chains))
    protein = example_protein.get(name="heterodimer")

    # Chain extraction should return used and unused chains
    p_used, p_unused = extract_chains(protein, sel_chains)
    assert np.unique(p_used.chain_index).tolist() == used_chains
    assert np.unique(p_unused.chain_index).tolist() == unused_chains


@pytest.mark.parametrize("name", ["heterodimer", "homodimer"])
def test_concat_proteins(example_protein, name):
    protein = example_protein.get(name=name)
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


@pytest.mark.parametrize("name", ["monomer", "heterodimer", "homodimer"])
def test_get_core_mask(example_protein, name):
    protein = example_protein.get(name=name)
    # Core mask should be boolean and have same length as number of residues
    core_mask = get_core_mask(protein)
    assert isinstance(core_mask, np.ndarray)
    assert core_mask.dtype == bool
    assert len(core_mask) == protein.n_res
    # Core mask should be deterministic
    core_mask_2 = get_core_mask(protein)
    assert np.array_equal(core_mask, core_mask_2)


def test_validate_chain(example_protein):
    protein = example_protein.get(name="heterodimer")
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


def test_validate_res(example_protein):
    protein = example_protein.get(name="heterodimer")
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
    anchor_res = "B5"
    res_array = validate_residues(protein, anchor_res, mode="anchor")
    assert len(res_array) == len(anchor_res.split(","))
    # Should throw an error if nonpolar residues are included
    anchor_res = "A1,B3,B5"
    with pytest.raises(ValueError):
        validate_residues(protein, anchor_res, mode="anchor")


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


guide_seqs = [
    ["X", "X", "X"], # No seq cond
    ["R", "R", "T"], # Full seq cond
    ["E", "H", "X"], # Partial seq cond
    ["E|D", "H", "X"], # Ambiguous (either-or) seq cond
    ["E|D|T", "H|K", "K"] # Mixed seq cond
]
x = 1 / 11. # Default ambiguous probability value (11 polar restypes)
y = x / 3. # Adjusted ambiguous probability value for one of three residues
seq_cond_values = [
    {
        "A": 0., "R": x, "N": x, "D": x, "C": 0., 
        "Q": x, "E": x, "G": 0., "H": x, "I": 0., 
        "L": 0., "K": x, "M": 0., "F": 0., "P": 0., 
        "S": x, "T": x, "W": x, "Y": x, "V": 0., 
    }, # No seq cond
    {
        "A": 0., "R": 2/3., "N": 0., "D": 0., "C": 0., 
        "Q": 0., "E": 0., "G": 0., "H": 0., "I": 0., 
        "L": 0., "K": 0., "M": 0., "F": 0., "P": 0., 
        "S": 0., "T": 1/3., "W": 0., "Y": 0., "V": 0., 
    }, # Full seq cond
    {
        "A": 0., "R": y, "N": y, "D": y, "C": 0., 
        "Q": y, "E": y + (1/3.), "G": 0., "H": y + (1/3.), "I": 0., 
        "L": 0., "K": y, "M": 0., "F": 0., "P": 0., 
        "S": y, "T": y, "W": y, "Y": y, "V": 0., 
    }, # Partial seq cond
    {
        "A": 0., "R": y, "N": y, "D": y + (1/6.), "C": 0., 
        "Q": y, "E": y + (1/6.), "G": 0., "H": y + (1/3.), "I": 0., 
        "L": 0., "K": y, "M": 0., "F": 0., "P": 0., 
        "S": y, "T": y, "W": y, "Y": y, "V": 0., 
    }, # Ambiguous (either-or) seq cond
]

@pytest.mark.parametrize("guide_seq,dist_values", zip(guide_seqs, seq_cond_values))
def test_seq_cond(guide_seq, dist_values):
    def get_expected_dist(dist_values):
        expected_dist = np.zeros(rc.restype_num + 1) # [21]
        for restype in rc.restypes:
            expected_dist[rc.restype_order[restype]] = dist_values[restype]
        return expected_dist
    expected_dist = get_expected_dist(dist_values)
    # Seq cond has a row for every residue, so we must aggregate it to check
    seq_cond = get_seq_cond_inf(guide_seq) # [1, 21, 3]
    seq_cond = np.squeeze(np.sum(seq_cond, axis=-1) / np.sum(seq_cond, axis=(1, 2))) # Renormalize
    assert_allclose(np.sum(seq_cond), 1., atol=1e-3)
    assert_allclose(seq_cond, expected_dist, atol=1e-5)

    # Check that omit_AA moves prob mass from omitted restypes to others
    omit_AA = ["N", "Q"]
    seq_cond_omit_AA = get_seq_cond_inf(guide_seq, omit_AA=omit_AA)
    seq_cond_omit_AA = np.squeeze(np.sum(seq_cond_omit_AA, axis=-1) / np.sum(seq_cond_omit_AA, axis=(1, 2))) # Renormalize
    assert (seq_cond_omit_AA[rc.restype_order["N"]] == 0.) and  (seq_cond_omit_AA[rc.restype_order["Q"]] == 0.)
    assert_allclose(np.sum(seq_cond_omit_AA), 1., atol=1e-3)


@pytest.mark.parametrize("non_net", ["G", "R"])
def test_get_network_res(example_protein, non_net):
    protein = example_protein.get(name="homodimer")
    # Full protein should return all non-Gly res
    net_res = get_network_res(protein, non_net=non_net) 
    non_net_res = np.where(protein.aatype != rc.restype_order[non_net])[0]   
    assert isinstance(net_res, str)
    net_res = net_res.split(":")
    assert len(net_res) == non_net_res.size
    assert len(net_res) != protein.n_res

    # All-N protein should return no hits
    protein.clear_sequence()
    protein.aatype[:] = rc.restype_order[non_net]
    net_res = get_network_res(protein, non_net=non_net)
    non_net_res = np.where(protein.aatype != rc.restype_order[non_net])[0]   
    assert net_res == ""
    assert len(net_res) == non_net_res.size
    assert len(net_res) == 0

    # All-A protein should return all residues as hits
    protein.aatype[:] = rc.restype_order["A"]
    net_res = get_network_res(protein, non_net=non_net)
    net_res = net_res.split(":")
    assert len(net_res) == protein.n_res


def test_get_symmetry_mask(example_protein):

    protein = example_protein.get(name="heterodimer")
    # Should throw ValueError if chains are different sizes
    with pytest.raises(ValueError):
        get_symmetry_mask(protein, "A,B")
    # Should throw ValueError if specified chains don't exist
    with pytest.raises(ValueError):
        get_symmetry_mask(protein, "A,C")
    # Should pass and return identity matrix if no chains specified
    symm_mask = get_symmetry_mask(protein, None)
    assert isinstance(symm_mask, np.ndarray)
    assert symm_mask.shape == (protein.n_res, protein.n_res)
    assert_allclose(symm_mask, np.eye(protein.n_res))

    protein = example_protein.get(name="homodimer")
    # Symmetry mask is non-identity and should be symmetric with a constant factor
    symm_mask = get_symmetry_mask(protein, "A,B")
    assert isinstance(symm_mask, np.ndarray)
    assert symm_mask.shape == (protein.n_res, protein.n_res)
    assert_allclose(symm_mask, symm_mask.T)
    assert not np.allclose(symm_mask, np.eye(protein.n_res))
    symm_factor_0 = np.sum(symm_mask, axis=0).astype(int)
    symm_factor_1 = np.sum(symm_mask, axis=1).astype(int)
    assert_allclose(symm_factor_0, 2)
    assert_allclose(symm_factor_1, 2)


@pytest.mark.parametrize("name", ["monomer", "heterodimer", "homodimer"])
def test_get_symmetry_index_asymm(example_protein, name):
    # Asymm idx should be monotonically increasing
    protein = example_protein.get(name=name)
    symm_mask = get_symmetry_mask(protein, None)
    symm_idx = get_symmetry_idx(symm_mask)
    assert isinstance(symm_idx, np.ndarray)
    assert symm_idx.shape == (protein.n_res,)
    assert_allclose(symm_idx, np.arange(protein.n_res))


def test_get_symmetry_index_symm(example_protein):
    # Symmetric idx should have repeated values tied across chains
    protein = example_protein.get(name="homodimer")
    symm_mask = get_symmetry_mask(protein, "A,B")
    symm_idx = get_symmetry_idx(symm_mask)
    assert not np.allclose(symm_idx, np.arange(protein.n_res))
    n_unique_vals = len(np.unique(symm_idx))
    symm_factor = len(symm_idx) // n_unique_vals
    assert symm_factor == 2
    assert n_unique_vals == protein.n_res // symm_factor
    chain_A_idx = symm_idx[np.where(protein.chain_index == 0)[0]]
    chain_B_idx = symm_idx[np.where(protein.chain_index == 1)[0]]
    assert_allclose(chain_A_idx, chain_B_idx)
    assert_allclose(chain_A_idx, np.arange(n_unique_vals))
    

def test_symmetrize_output(example_protein):
    protein = example_protein.get(name="homodimer")
    protein_copy = example_protein.get(name="homodimer")
    protein.clear_sequence()

    # This network should symmetrize without issues
    net_pos = np.array([77, 88, 72 + 208]) # [A79Y, A90D, B74Y]
    protein.aatype[:] = rc.restype_order["G"]
    protein.aatype[net_pos] = protein_copy.aatype[net_pos]
    protein.atom27_xyz[net_pos] = protein_copy.atom27_xyz[net_pos]
    protein.atom27_mask[net_pos] = protein_copy.atom27_mask[net_pos]

    df = pd.DataFrame({"protein": [protein], 
        "network": [get_network_res(protein)], 
        "buried_heavy_unsats": [1], 
        "buried_unsat_Hpol": [0], 
        "HB_Score_full": [3.3], 
        "HB_Score_hb": [-0.251]
        })

    init_row = df.iloc[0]
    row = symmetrize_output(row=0, df=df, symm_mask=get_symmetry_mask(protein, "A,B"))
    assert row is not None
    # Check that metrics were updated
    for value in ["buried_heavy_unsats", "buried_unsat_Hpol", "HB_Score_full", "HB_Score_hb"]:
        assert value in row.keys()
        assert row[value] == (2 * init_row[value])

    # Check that new protein actually has network symmetrized
    new_protein = row.protein
    old_net_res = np.where(protein.aatype != rc.restype_order["G"])[0]
    new_net_res = np.where(new_protein.aatype != rc.restype_order["G"])[0]
    assert len(new_net_res) == 2 * len(old_net_res)
    assert np.all(np.isin(old_net_res, new_net_res))

    # New atom mask should have new atoms added that weren't in the old protein
    new_net_res_only = np.setdiff1d(new_net_res, old_net_res)
    old_atom_mask = np.sum(protein.atom27_mask[new_net_res_only, 4:], axis=-1)
    new_atom_mask = np.sum(new_protein.atom27_mask[new_net_res_only, 4:], axis=-1)
    assert not np.allclose(old_atom_mask, new_atom_mask)
    assert np.all(new_atom_mask > old_atom_mask)

    # If we simulate clashing with a really big clash threshold, it should fail and return None
    row = symmetrize_output(row=0, df=df, symm_mask=get_symmetry_mask(protein, "A,B"), clash_thresh=10.)
    assert row is None

    # If we give it a quasi-symmetric network, it should fail and return None
    # In this context, quasi-symmetric means that symmetry-mates are assigned different restypes
    # e.g., A79R and B79S can't be reconciled
    protein.aatype[new_net_res_only] = rc.restype_order["R"]
    df.protein = protein
    df.network = get_network_res(protein)
    row = symmetrize_output(row=0, df=df, symm_mask=get_symmetry_mask(protein, "A,B"))
    assert row is None

