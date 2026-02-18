import pytest 
from pathlib import Path
import os 
import numpy as np
from numpy.testing import assert_allclose

from hbdesigner.data.protein import Protein
from hbdesigner.data.features import (
    calc_sc_dihedrals, 
    build_sc_from_chi, 
    calc_bb_dihedrals, 
    sincos_to_angle
)
from hbdesigner.data.hbnet import crop_by_distance, crop_around_network


@pytest.fixture
def example_protein() -> Protein:
    """Enables multiple protein instances to be generated."""

    class ProteinFactory(object):
        def get(self, discard_Hs=False, name="monomer") -> Protein:
            fpath = {
                "monomer": "examples/monomer/1PGA.pdb",
                "heterodimer": "examples/interface/1YRK.pdb",
                "homodimer": "examples/interface/symm_lazy/10GS.pdb",
            }
            pdb_file = os.path.join(
                Path(__file__).parents[1], fpath[name]
            )
            return Protein.from_pdb_file(pdb_file, discard_Hs=discard_Hs)

    return ProteinFactory()


def test_protein_loading(example_protein):
    # Check loading with/without H atoms works
    protein = example_protein.get(discard_Hs=True)
    h_atoms = np.sum(protein.atom27_mask[:, 14:], axis=1)
    assert np.all(h_atoms == 0)

    protein = example_protein.get(discard_Hs=False)
    h_atoms = np.sum(protein.atom27_mask[:, 14:], axis=1)
    assert np.all(h_atoms > 0)


def test_dihedral_calc(example_protein):
    # Check that sidechain idealization doesn't break the backbone or lose atoms
    protein = example_protein.get()
    chi_angles, chi_angle_mask = calc_sc_dihedrals(
        protein.atom27_xyz,
        protein.aatype,
    )
    assert chi_angles.shape == (protein.n_res, 4)
    assert chi_angle_mask.shape == (protein.n_res, 4)
    atom14_xyz, atom14_mask = build_sc_from_chi(
        protein.atom27_xyz[:, :4],
        protein.aatype,
        chi_angles=chi_angles,
        chi_angle_mask=chi_angle_mask,
    )
    assert atom14_xyz.shape == (protein.n_res, 14, 3)
    assert atom14_mask.shape == (protein.n_res, 14)
    assert np.all(atom14_mask == protein.atom27_mask[:, :14])
    # Sidechain atoms may move slightly due to idealization, but backbone should be identical
    assert np.all(atom14_xyz[:, :4] == protein.atom27_xyz[:, :4])

    # Check that backbone dihedral calculation returns expected shapes and masks
    bb_dihedrals, bb_dihedral_mask = calc_bb_dihedrals(
        protein.atom27_xyz,
        protein.residue_index,
        return_mask=True,
    )
    assert bb_dihedrals.shape == (protein.n_res, 3)
    assert bb_dihedral_mask.shape == (protein.n_res, 3)

    # Check that angle sincos conversion is consistent
    sin_cos = np.stack((np.sin(bb_dihedrals), np.cos(bb_dihedrals)), axis=-1)
    bb_dih_out = sincos_to_angle(sin_cos)
    assert_allclose(bb_dihedrals, bb_dih_out)


def test_protein_funcs(example_protein):
    protein = example_protein.get()

    # clear_sidechains should drop sc atoms but not aatype
    mask_before = np.copy(protein.atom27_mask)
    xyz_before = np.copy(protein.atom27_xyz)
    aatype_before = np.copy(protein.aatype)
    protein.clear_sidechains()
    # Shouldn't change backbone or aatypes
    assert_allclose(mask_before[:, :4], protein.atom27_mask[:, :4])
    assert_allclose(xyz_before[:, :4, :], protein.atom27_xyz[:, :4, :])
    assert_allclose(aatype_before, protein.aatype)
    # Should drop sidechain atoms and masks
    assert_allclose(protein.atom27_mask[:, 4:], 0)
    assert_allclose(protein.atom27_xyz[:, 4:, :], 0.0)

    # clear_sequence should drop sc atoms and aatype
    protein = example_protein.get()
    mask_before = np.copy(protein.atom27_mask)
    xyz_before = np.copy(protein.atom27_xyz)
    aatype_before = np.copy(protein.aatype)
    protein.clear_sequence()
    # Shouldn't change backbone or aatypes
    assert_allclose(mask_before[:, :4], protein.atom27_mask[:, :4])
    assert_allclose(xyz_before[:, :4, :], protein.atom27_xyz[:, :4, :])
    # All aatypes are now UNK (20)
    assert_allclose(np.full_like(protein.aatype, fill_value=20), protein.aatype)
    # Should drop sidechain atoms and masks
    assert_allclose(protein.atom27_mask[:, 4:], 0)
    assert_allclose(protein.atom27_xyz[:, 4:, :], 0.0)
    assert np.all(protein.aatype != aatype_before)

    # Neighbor mask should return more hits with larger distance cutoff
    copy_protein = example_protein.get()
    protein.set_neighbor_mask(dist=5.0)
    copy_protein.set_neighbor_mask(dist=10.0)
    assert np.sum(protein.neighbor_mask) < np.sum(copy_protein.neighbor_mask)


def test_protein_hash(example_protein):

    # Hash and Eq methods enable comparison of Protein objects
    p1 = example_protein.get()
    p2 = example_protein.get()
    assert p1 == p2

    # Changes to any part of the structure should break equality
    p2.clear_sidechains()
    assert p1 != p2

    p2 = example_protein.get()
    p2.b_factors[:] = 100
    assert p1 != p2

    # Uniqueness relies on hashing
    p_list = [p1, example_protein.get(), p2]
    assert len(set(p_list)) == 2


def test_crop(example_protein):
    protein = example_protein.get()
    # Cropped protein should have k residues
    k = protein.n_res // 2
    cropped = protein.crop(topk=k)
    assert (cropped.n_res == k) and (cropped.n_res != protein.n_res)

    # If crop > protein, it can be padded
    cropped = protein.crop(topk=protein.n_res + 10, pad=True)
    assert cropped.n_res == protein.n_res + 10
    cropped = protein.crop(topk=protein.n_res + 10, pad=False)
    assert cropped.n_res == protein.n_res

    # Repeated cropping shouldn't drop residues
    cropped = protein.crop(k).crop(k)
    assert cropped.n_res == k

    # Crop by distance should return a smaller protein
    cropped, _ = crop_by_distance(protein, net_pos=np.array([14]), d=10.0)
    assert cropped.n_res < protein.n_res
    cropped_2, _ = crop_by_distance(protein, net_pos=np.array([14]), d=5.0)
    assert cropped_2.n_res < cropped.n_res

    # Mask function should keep only specified residues
    keep_res = np.arange(0, 10)
    masked = protein.mask(keep_res)
    assert masked.n_res == 10
    # Residues are 1-indexed in Protein
    assert_allclose(masked.residue_index, keep_res + 1)

    # crop_around_network should find the KNN of specified residues
    net_pos = np.array([4, 29, 33])
    k = 16
    cropped, crop_idxs = crop_around_network(
        protein, 
        net_pos=net_pos, 
        topk=k
    )
    assert cropped.n_res == k
    assert len(crop_idxs) == k
    for net_res in net_pos:
        assert net_res + 1 in cropped.residue_index

