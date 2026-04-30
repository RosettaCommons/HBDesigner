import argparse
import os
from typing import Sequence


class FileArgumentParser(argparse.ArgumentParser):
    """Overwrites default ArgumentParser to better handle flag files."""

    def convert_arg_line_to_args(self, arg_line: str) -> Sequence[str]:
        """Read from files where each line contains a flag and its value, e.g.
        '--flag value'. Also safely ignores comments denotes with '#' and
        empty lines.
        """

        # Remove any comments from the line
        arg_line = arg_line.split("#")[0]

        # Escapte if the line is empty
        if not arg_line:
            return None

        # Separate flag and values
        split_line = arg_line.strip().split(" ")

        # If there is actually a value, return the flag value pair
        if len(split_line) > 1:
            return [split_line[0], " ".join(split_line[1:])]
        # Return just flag if there is no value
        else:
            return split_line


def get_hbdes_parser() -> FileArgumentParser:
    parser = FileArgumentParser(
        description="Required and optional arguments for running HBDesigner. For help on individual arguments, run with the --help flag or refer to the documentation."
    )

    # Filepaths
    parser.add_argument(
        "--pdb", type=str, required=True, help="Relative file path and name of the PDB file to process."
    )
    parser.add_argument(
        "--design_model",
        type=str,
        required=False,
        default="design_020",
        choices=["design_002", "design_020"],
        help="Design model to use. Default is 'design_020' (moderate noise), but 'design_002' (low noise) is also available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference only on CPU by loading CPU-specific YAML configs. This is not recommended, as it will be much slower than running on GPU, but is available for users without access to a CUDA-enabled GPU. By default, the script will attempt to run on GPU if available.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default=os.getcwd(),
        help="Relative path to output directory for saving new files. Defaults to current working directory. If the given directory does not exist, it will be created. Output files include designed HBNet PDB files and a summary CSV file with design metrics for each network.",
    )
    # Runtime optimization params
    parser.add_argument(
        "--n_workers",
        type=int,
        required=False,
        default=1,
        help="Workers for parallelization (packing). Default is 1. More workers will speed up predictions. The number available will depend on your system.",
    )
    # Sampling params
    parser.add_argument(
        "--n_samples",
        type=int,
        required=False,
        default=100,
        help="Number of unique samples to generate. Default is 100. More samples will increase diversity but also increase inference time.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=5,
        help="How many unique samples to keep after ranking. Default is 5.",
    )
    parser.add_argument(
        "--n_res",
        type=int,
        required=False,
        choices=range(2, 7),
        default=2,
        help="Size of desired HBNet, in residues. Default is 2. Valid range is 2-6 (inclusive).",
    )
    parser.add_argument(
        "--T_range",
        type=float,
        required=False,
        nargs=2,
        default=[0.1, 1.0],
        help="Temperature range for sampling. Default is [0.1, 1.0]. Lower temperatures will yield more conservative designs, while higher temperatures will yield more diverse designs.",
    )
    parser.add_argument(
        "--min_burial",
        type=float,
        required=False,
        default=0.0,
        help="Minimum burial for designable positions, as calculated by Rosetta's sidechain neighbor algorithm. Defaults to 0.0. See https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/ResidueSelectors/ResidueSelectors#residueselectors_conformation-dependent-residue-selectors_layerselector or https://docs.rosettacommons.org/docs/latest/rosetta_basics/scoring/BuriedUnsatPenalty#algorithm for more information.",
    )
    # Conditional information
    parser.add_argument(
        "--guide_res",
        type=str,
        required=False,
        default=None,
        help="Guide residues. Off by default. Model will calculate Cb-centroid of these residues and place the guide atom near the centroid."
        "Uses PDB chain/resnum format. Example: 'A12,B13,B49' ",
    )
    parser.add_argument(
        "--guide_radius",
        type=float,
        required=False,
        default=1e6,
        help="Hard constraint on designable positions based on Cb distance "
        "(Angstrom) from guide atom. Off by default.",
    )
    parser.add_argument(
        "--guide_seq",
        type=str,
        required=False,
        default=None,
        help="Guide sequence. Default is 'X,X,X' (3 unknowns). Options include 'S,T' (one SER, one THR), 'T,X' (one THR, one UNK), 'S|T,N' (one SER or THR, one ASN) etc.",
    )
    # Scoring params
    parser.add_argument(
        "--max_BUNs",
        type=int,
        required=False,
        default=0,
        help="Maximum buried unsatisfied hydrogen atoms (BUNs, buried but not participating in a hydrogen bond) to allow in the network. Default is 0. Raising this will make filtering more permissive. ",
    )
    parser.add_argument(
        "--max_BUPHs",
        type=int,
        required=False,
        default=5,
        help="Maximum buried unsatisfied polar hydrogen atoms (BUPHs, buried but not participating in a hydrogen bond) to allow in the network. Default is 5. Raising this will make filtering more permissive. ",
    )
    parser.add_argument(
        "--min_sat",
        type=float,
        required=False,
        default=0.5,
        help="Minimum saturation score to allow in the network. Default is 0.5. Raising this will make filtering more strict. ",
    )
    parser.add_argument(
        "--max_hb_energy",
        type=float,
        required=False,
        default=0.0,
        help="Maximum hydrogen bond energy to allow in the network. Default is 0.0. Raising this will make filtering more permissive.",
    )
    # Parsing params
    parser.add_argument(
        "--symm_chains",
        type=str,
        required=False,
        default=None,
        help="Option to symmetrize output networks for convenience. Specify symmetric chains as 'A,B;C,D' to symmetrize A with B and C with D. If you wanted to symmetrize all four together, the input would be 'A,B,C,D'. By default, no symmetrization will be performed. ",
    )
    parser.add_argument(
        "--symm_file", 
        type=str,
        required=False,
        default=None,
        help="Symmetry file (.symm) to use for symmetrization. Only used for strict symmetry. You can use Rosetta's make_symmdef_file.pl script to generate a .symm file from your PDB."
    )
    parser.add_argument(
        "--sel_chains",
        required=False,
        type=str,
        default=None,
        help="Option to select specific chain(s) to run HBDesigner on. Format: 'A,C'. Off by default (will use all chains). This option removes chains from the input file *before* they are passed to HBDesigner. HBDesigner will then graft chain B back in so the output still contains both chains.",
    )
    parser.add_argument(
        "--min_core_res",
        required=False,
        type=int,
        default=0,
        help="Minimum number of core residues required for each network. Defaults to 0.",
    )
    parser.add_argument(
        "--anchor_res", 
        required=False, 
        type=str,
        default=None,
        help="Comma-separated list of residues to use as anchor residues during design, in PDB chain/resnum format. Example: 'A12,B13,B49'."
    )
    parser.add_argument(
        "--max_hb_score", 
        required=False, 
        type=float, 
        default=0.0, 
        help="Maximum energy score to allow for returned networks. More negative values are more strict. Default is 0.0"
    )
    parser.add_argument(
        "--omit_chains", 
        required=False, 
        type=str,
        default=None,
        help="Comma-separated list of chains to omit from design. Residues in these chains will not be eligible for use in designed networks. Example: A,C. This option allows HBDesigner to see all of the chains (even the omitted ones) but prevents HBDesigner from using residues in the omitted chains to form the hydrogen bonding network. This is different from the --sel_chains option, which removes the omitted chains from the input file entirely (i.e. HBDesigner will not even see the omitted chains)."
    )
    parser.add_argument(
        "--omit_AA",
        required=False,
        type=str,
        default=None,
        help="Comma-separated list of amino acids to omit from design. Residues of these types will not be eligible for use in designed networks. Example: R,K means no LYS or ARG allowed."
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=None,
        help="Random seed for reproducible sampling. Default is no seed. Note that due to the use of parallelization and Rosetta scoring, results may not be exactly reproducible even with a set seed. Running with parallelization turned off ('n_workers' set to 0) will increase the likelihood of similar results between runs, but small energy changes will still be seen from Rosetta.",
    )
    parser.add_argument(
        "--design_model_ckpt", 
        required=False,
        type=str,
        default=None,
        help="Path to custom design model checkpoint. If not specified, will use default checkpoint for the specified design model (e.g. 'design_020')."
    )
    parser.add_argument(
        "--packing_model_ckpt",
        required=False,
        type=str,
        default=None,
        help="Path to custom packing model checkpoint. If not specified, will use default checkpoint."
    )

    return parser
