import subprocess
import sys
import os
import shutil
import pytest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (TEST_DIR / "..").resolve()

RUN_COMMAND=shutil.which("/home/woodsh/HBDesigner/.venv/bin/run_hbdesigner")

SCENARIOS = [
    (
        "monomer_unconditional",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
        ],
    ),
    (
        "guide_atom_conditioning",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
            "--guide_res", "A3,A26"
        ],
    ),
    (
        "omit_AA",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
            "--omit_AA", "R,K"
        ],
    ),
    (
        "sequence_conditioning_ambiguous",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "3",
            "--guide_seq", "S,N|Q,T"
        ],
    ),
    (
        "sequence_conditioning_full",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "3",
            "--guide_seq", "S,N,T"
        ],
    ),
    (
        "sequence_conditioning_partial",
        [
            "--pdb", "examples/monomer/1PGA.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "3",
            "--guide_seq", "X,T,X"
        ],
    ),
    (
        "interface_one_sided",
        [
            "--pdb", "examples/interface/1YRK.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
            "--top_k", "1",
            "--anchor_res", "B5",
            "--omit_chains", "B"
        ],
    ),
    (
        "interface_symm_lazy",
        [
            "--pdb", "examples/interface/symm_lazy/10GS.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
            "--symm_chains", "A,B"
        ],
    ),
    (
        "interface_symm_strict",
        [
            "--pdb", "examples/interface/symm_strict/5J0K_clean_symm.pdb",
            "--n_workers", "4",
            "--n_samples", "10",
            "--n_res", "2",
            "--symm_chains", "A,B",
            "--symm_file", "examples/interface/symm_strict/5J0K.symm"
        ],
    ),
]

@pytest.mark.parametrize("name, args", SCENARIOS)
def test_scenarios(name, args):
    """
    Runs hbdesigner via subprocess for each scenario
    """
    cmd = [RUN_COMMAND, *args]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    # Debug info if something fails
    if result.returncode != 0:
        print(f"\n=== Scenario {name} failed ===")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0
