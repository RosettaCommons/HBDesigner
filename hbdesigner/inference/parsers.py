import argparse
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
        description="Parser for HBDesigner inference config files."
    )

    # Filepaths
    parser.add_argument(
        "--pdb", type=str, required=True, help="Input PDB file to process."
    )
    parser.add_argument(
        "--design_ckpt",
        type=str,
        required=True,
        help="Design model checkpoint file to use for inference.",
    )
    parser.add_argument(
        "--design_cfg",
        type=str,
        required=True,
        help="Design model config file to use for model params.",
    )
    parser.add_argument(
        "--pack_cfg",
        type=str,
        default=None,
        help="Config file for packing model. Only needed if --packer is 'hbdes3'.",
    )
    parser.add_argument(
        "--pack_ckpt",
        type=str,
        default=None,
        help="Checkpoint for packing model. Only needed if --packer is 'hbdes3'.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default=None,
        help="Output directory for saving new files. Defaults to None, which means no files will be saved.",
    )
    # Runtime optimization params
    parser.add_argument(
        "--n_workers",
        type=int,
        required=False,
        default=1,
        help="Workers for parallelization (packing). Default is 1. More workers will speed up predictions.",
    )
    parser.add_argument(
        "--packer",
        type=str,
        required=False,
        default="hbpacker",
        choices=["hbpacker", "rosetta", "pippack", "none"],
        help="Packer to use. Default is 'hbpacker', but 'rosetta', 'pippack', and 'none' are also available.",
    )
    parser.add_argument(
        "--pack_crop",
        type=float,
        required=False,
        default=10.0,
        help="For speed, packing only uses residues within this many Angstrom of the designed network. Default is 10 Angstrom. To disable cropped packing, set this to 0.",
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
        help="Size of desired HBNet, in residues. Default is 2. Valid range is 2-6.",
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
        help="Minimum burial for designable positions, as calculated by Rosetta's sidechain neighbor algorithm. Defaults to 0.0.",
    )
    parser.add_argument(
        "--bb_noise",
        type=float,
        required=False,
        default=0.0,
        help="Level of backbone noise to add for inference. More noise means more diversity, but lower avg success rate. Default is 0.0.",
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
        help="Guide sequence. Default is 'XXX' (3 unknowns). Options include 'ST' (one SER, one THR), 'SX' (one SER, one UNK), etc.",
    )
    # Scoring params
    parser.add_argument(
        "--max_BUNs",
        type=int,
        required=False,
        default=0,
        help="Maximum buried unsats to allow in the network. Default is 0. Raising this will make filtering more permissive. ",
    )
    parser.add_argument(
        "--max_BUPHs",
        type=int,
        required=False,
        default=5,
        help="Maximum buried unsat polar Hs to allow in the network. Default is 5. Raising this will make filtering more permissive. ",
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
        help="Maximum h-bond energy to allow in the network. Default is 0.0. Raising this will make filtering more permissive.",
    )
    # Parsing params
    parser.add_argument(
        "--symm_chains",
        type=str,
        required=False,
        default=None,
        help="Option to symmetrize output networks for convenience. Specify symmetric chains as 'A,B;C,D' etc.",
    )
    parser.add_argument(
        "--sel_chains",
        required=False,
        type=str,
        default=None,
        help="Option to select specific chain(s) to run HBDesigner on. Format: 'A,C'. Off by default (will use all chains).",
    )
    parser.add_argument(
        "--min_core_res",
        required=False,
        type=int,
        default=0,
        help="Minimum number of core residues required for each network. Defaults to 0.",
    )
    parser.add_argument(
        "--fixed_res", 
        required=False, 
        type=str,
        default=None,
        help="Comma-separated list of residues to keep fixed during design, in PDB chain/resnum format. Example: 'A12,B13,B49'."
    )

    return parser
