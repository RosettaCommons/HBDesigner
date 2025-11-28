from hbdesigner.data.protein import Protein
from typing import List
from multiprocessing import Pool
from functools import partial
import hbdesigner.data.residue_constants as rc
import numpy as np 

"""
OpenMM functions adapted from FreeBindCraft (https://github.com/cytokineking/FreeBindCraft/)
"""

import openmm
from openmm import app, unit, Platform, OpenMMException, CustomHbondForce
from pdbfixer import PDBFixer
import time
import io
import gc


# Cache a single OpenMM ForceField instance to avoid repeated XML parsing per relaxation
_OPENMM_FORCEFIELD_SINGLETON = None

def _get_openmm_forcefield():
    global _OPENMM_FORCEFIELD_SINGLETON
    if _OPENMM_FORCEFIELD_SINGLETON is None:
        # _OPENMM_FORCEFIELD_SINGLETON = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        _OPENMM_FORCEFIELD_SINGLETON = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml') # best
        # _OPENMM_FORCEFIELD_SINGLETON = app.ForceField('amber99sb.xml', 'implicit/gbn2.xml')
        # _OPENMM_FORCEFIELD_SINGLETON = app.ForceField('charmm36.xml', 'implicit/obc2.xml')
    return _OPENMM_FORCEFIELD_SINGLETON


def run_openmm_minimize(p: Protein) -> Protein:
    """Run OpenMM minimization on a single Protein object.
    """
    try:
        t0 = time.time()
        pdb_str = p.to_pdb(unk_to_gly=True)
        
        # Do quick-and-dirty hbond check before committing to a slow minimization
        from hbdesigner.data.hbnet import run_reduce, biotite_hbond_detect
        pdb_str = run_reduce(pdb_str, his=True, flip=False)
        # p = Protein.from_pdb_string(pdb_str, discard_Hs=False)
        # pair_list, stats = biotite_hbond_detect(p, AH_dist=3.5, AHD_angle=90.)

        # import networkx as nx
        # g = nx.Graph()
        # g.add_edges_from(pair_list)
        # n_nodes = g.number_of_nodes()
        # n_expected_nodes = np.where(p.aatype != rc.restype_order["G"])[0].size

        # # Return early, if it fails completely
        # if n_nodes < n_expected_nodes:
        #     print("quick-and-dirty check failed, skipping minimization")
        #     return p

        # Load into OpenMM via PDBFixer, add hydrogens
        buffer = io.StringIO(pdb_str)
        fixer = PDBFixer(pdbfile=buffer)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        # Can provide forcefield here to do proper optH instead of full minimize
        forcefield = _get_openmm_forcefield()
        fixer.addMissingHydrogens(pH=7.0, forcefield=forcefield)

        # print(fixer.topology.getNumAtoms(), "atoms in OpenMM system for minimization")
        # Configure OpenMM minimization
        system = forcefield.createSystem(fixer.topology,
                                            nonbondedMethod=app.CutoffNonPeriodic,
                                            nonbondedCutoff=1.0*unit.nanometer)

        integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin,
                                                        1.0/unit.picosecond,
                                                        0.002*unit.picoseconds)

        # Platform selection
        plat_used = None
        props = {}
        sim = None
        max_iterations = 300 # 1000 default
        force_tolerance_kj_mol_nm = 2.0 # 0.1 default
        # NOTE: which platform to use?
        # OpenCL and CUDA are GPU-based, can't parallelize though
        platform_order = ['CPU']
        # platform_order = ['OpenCL']
        # platform_order = ['CUDA']
        for p_name in platform_order:
            try:
                platform_obj = Platform.getPlatformByName(p_name)
                if p_name == 'CUDA':
                    props = {'CudaPrecision': 'mixed'}
                elif p_name == 'OpenCL':
                    props = {'OpenCLPrecision': 'single'}
                sim = app.Simulation(fixer.topology, system, integrator, platform_obj, props)
                plat_used = p_name
                break
            except Exception:
                sim = None
                continue
        if sim is None:
            raise OpenMMException("No suitable OpenMM platform for minimization")

        # print(plat_used, "platform selected for OpenMM minimization")
        sim.context.setPositions(fixer.positions)
        tol = force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer
        
        # Fix all atoms except network sidechains
        for residue in sim.topology.residues():
            for atom in residue.atoms():
                if atom.name in {'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3'} or residue.name == "GLY":
                    system.setParticleMass(atom.index, 0)

        sim.minimizeEnergy(tolerance=tol, maxIterations=max_iterations)
        positions = sim.context.getState(getPositions=True).getPositions()

        # Export from OpenMM back to Protein
        buffer = io.StringIO()
        app.PDBFile.writeFile(sim.topology, positions, buffer, keepIds=True)
        p = Protein.from_pdb_string(buffer.getvalue(), discard_Hs=False)

        # with open("openMM.pdb", "w") as fopen:
        # # with open("REDUCE_openMM.pdb", "w") as fopen:
        #     fopen.write(p.to_pdb(unk_to_gly=True))
        # quit()

        # Cleanup
        try:
            del sim, integrator, system, fixer
        except Exception:
            pass
        gc.collect()
        t1 = time.time()
        # print(t1 - t0, '\t...minimizedtime')

        if hasattr(p, "pack_time"):
            p.pack_time = p.pack_time + (t1 - t0)
        else:
            p.pack_time = t1 - t0

    except Exception as e:
        print("OpenMM minimization failed:", e)

    return p


def run_openmm_optH(p: Protein) -> Protein:
    """Run OpenMM minimization on a single Protein object.
    """

    t0 = time.time()

    # Load into OpenMM via PDBFixer, add hydrogens
    pdb_str = p.to_pdb(unk_to_gly=True)
    buffer = io.StringIO(pdb_str)
    fixer = PDBFixer(pdbfile=buffer)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    # Can provide forcefield here to do proper optH instead of full minimize
    forcefield = _get_openmm_forcefield()
    fixer.addMissingHydrogens(pH=7.0, forcefield=forcefield)

    # Export from OpenMM back to Protein
    buffer = io.StringIO()
    app.PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
    p = Protein.from_pdb_string(buffer.getvalue(), discard_Hs=False)

    return p


def openmm_minimize(proteins: List[Protein], n_workers: int = 1) -> List[Protein]:
    """Minimize proteins using OpenMM in parallel.

    Args:
        p (List[Protein]): List of Protein objects to minimize.
        n_workers (int, optional): Number of parallel workers. Defaults to 1.

    Returns:
        List[Protein]: List of minimized Protein objects.
    """

    with Pool(n_workers) as p:
        proteins = p.map(
            partial(
                run_openmm_minimize,
            ),
            proteins,
        )

    # Compile all proteins into one big Protein batch w/100A offset
    # offset = 0.0
    # offset_chid = 0
    # import numpy as np
    # total_res = 0
    # total_ch = 0
    # chain_key = {}
    # for i, p in enumerate(proteins):
    #     n_chains = p.chain_index.max() + 1
    #     p.atom27_xyz[...,] += offset
    #     p.chain_index[...,] += offset_chid
    #     chain_key[i] = np.unique(p.chain_index)
    #     total_ch += np.unique(p.chain_index).size
    #     proteins[i] = p
    #     offset += 100.0
    #     total_res += p.n_res
    #     offset_chid += n_chains

    # from hbdesigner.inference.protein_ops import concat_proteins
    # p_all = concat_proteins(proteins)
    # print(len(proteins), total_ch, p_all.n_res, '***', total_res)

    # # with open("before.pdb", "w") as f:
    # #     f.write(p_all.to_pdb(unk_to_gly=True))
    # p_all = run_openmm_minimize(p_all)
    # # with open("after.pdb", "w") as f:
    # #     f.write(p_all.to_pdb(unk_to_gly=True))

    # for i, _ in enumerate(proteins):
    #     i_chains = chain_key[i]
    #     i_idx = np.isin(p_all.chain_index, i_chains)
    #     r_idx = np.where(i_idx)[0]
    #     p_i = p_all.mask(r_idx)
    #     proteins[i] = p_i

    return proteins
