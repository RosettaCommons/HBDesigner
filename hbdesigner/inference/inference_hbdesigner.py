import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Union
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

pd.options.mode.chained_assignment = None  # default='warn'

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import calc_sc_dihedrals, impute_CB, build_sc_from_chi
from hbdesigner.data.hbnet import (
    batch_to_proteins,
    initialize_rosetta,
    get_guide_atom,
)
from hbdesigner.data.protein import Protein
from hbdesigner.inference.parsers import get_hbdes_parser
from hbdesigner.inference.protein_ops import (
    extract_chains,
    get_core_mask,
    get_symmetry_mask,
    validate_residues,
    add_guide_atom,
    concat_proteins,
    symmetrize_output,
    get_network_res,
)
from hbdesigner.inference.packing import (
    pack_and_score_network,
    minimize_and_score_network,
)
from hbdesigner.model.hbdesign_model import HBDesigner, load_HBDesigner
from hbdesigner.model.hbpacker_model import load_HBPacker

from hbdesigner.model.pippack_model import (
    PIPPackFineTune,
    apply_logits_to_proteins,
    get_sidechain_logits,
    load_PIPPack,
)
from hbdesigner.scripts.train_hbdesigner import HBDesignerDataset
from hbdesigner.scripts.train_hbpacker import HBPackerDataset
from pyrosetta.rosetta.basic.options import set_real_option


class InferenceDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        scaffold: Protein,
        preds: List[Dict[str, Any]],
        batch_size: int,
        pack_crop: float,
    ):
        self.scaffold = scaffold
        self.preds = preds
        self.batch_size = batch_size
        self.pack_crop = pack_crop
        self.start, self.end = 0, len(self.preds)

    def __len__(self):
        return len(self.preds)

    def __iter__(self):
        # Each worker gets a different chunk of the dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                np.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            id = worker_info.id
            iter_start = self.start + id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.generate(iter_start, iter_end))

    def generate(self, start: int, end: int):
        """
        Yields batches of samples from a worker until finished.
        Called by self.__iter__().

        Arguments:
            start (int): Start index of this worker's chunk of the dataset.
            end (int): End index of this worker's chunk of the dataset.

        Returns:
            List(gd.Data): List of Data objects representing a batch before collation.

        """
        n_nodes = 0
        samples = []
        for idx in range(start, end + 1):
            # Build up batch to reach specified size
            try:
                p = self.preds[idx]
                sample = HBPackerDataset.featurize_inference(
                    deepcopy(self.scaffold),
                    np.array(p["net_res"]),
                    np.array(p["seq"]),
                    pack_crop=self.pack_crop,
                )
            except IndexError:
                sample = samples[-1]

            if (n_nodes + sample.num_nodes > self.batch_size) or (idx == end):
                yield samples
                n_nodes = 0
                samples = []
            n_nodes += sample.num_nodes
            samples.append(sample)


class HBDesRunner:
    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        self.validate_inputs()

        # Reconfigure design and packing configs
        self.design_cfg = OmegaConf.load(self.opts.design_cfg)
        self.pack_cfg = OmegaConf.load(self.opts.pack_cfg)

        # Make output directory
        if self.opts.out_dir is not None:
            os.makedirs(self.opts.out_dir, exist_ok=True)

    def validate_inputs(self):
        """
        Check provided user inputs to ensure they are valid.
        """
        # Filepaths
        assert os.path.isfile(self.opts.pdb), (
            f"ERROR: Invalid input file {self.opts.pdb} provided."
        )
        assert os.path.isfile(self.opts.design_ckpt), (
            f"ERROR: Invalid design ckpt file {self.opts.design_ckpt} provided."
        )
        assert os.path.isfile(self.opts.design_cfg), (
            f"ERROR: Invalid design config file {self.opts.design_cfg} provided."
        )
        assert os.path.isfile(self.opts.pack_ckpt), (
            f"ERROR: Invalid pack ckpt file {self.opts.pack_ckpt} provided."
        )
        assert os.path.isfile(self.opts.pack_cfg), (
            f"ERROR: Invalid pack config file {self.opts.pack_cfg} provided."
        )

        assert 1 <= self.opts.n_workers, (
            f"ERROR: Invalid number of workers {self.opts.n_workers} provided. --n_workers must be >=1."
        )

        # Sampling params
        assert 0 < self.opts.n_samples, (
            f"ERROR: Invalid number of samples {self.opts.n_samples} provided. --n_samples must be a positive integer."
        )
        assert 0 <= self.opts.top_k, (
            f"ERROR: Invalid top-k samples {self.opts.top_k} provided. --top_k cannot be negative."
        )
        assert 2 <= self.opts.n_res <= 6, (
            f"ERROR: Invalid number of residues {self.opts.n_res} provided. --n_res must be an integer in the range [2,6]."
        )
        valid_T_range = (
            (0.0 <= self.opts.T_range[0] <= 1.0)
            and (0.0 <= self.opts.T_range[1] <= 1.0)
            and (self.opts.T_range[0] <= self.opts.T_range[1])
        )
        assert valid_T_range, (
            f"ERROR: Invalid temperature range {self.opts.T_range} provided. --T_range must be a value in the range [0.0, 1.0] in ascending order."
        )
        assert 0.0 <= self.opts.min_burial, (
            f"ERROR: Invalid minimum burial {self.opts.min_burial} provided. --min_burial cannot be negative."
        )
        assert 0.0 <= self.opts.bb_noise, (
            f"ERROR: Invalid backbone noise {self.opts.bb_noise} provided. --bb_noise cannot be negative."
        )

        # Conditioning info
        # Can't validate --guide_res until PDB is loaded
        assert 0.0 < self.opts.guide_radius, (
            f"ERROR: Invalid guide radius {self.opts.guide_radius} provided. --guide_radius cannot be negative."
        )
        if self.opts.guide_seq is None:
            self.opts.guide_seq = "X" * self.opts.n_res
        assert len(self.opts.guide_seq) == self.opts.n_res, (
            f"ERROR: Length of guide seq ({len(self.opts.guide_seq)}, {self.opts.guide_seq}) must match number of designed residues ({self.opts.n_res})"
        )

        # Scoring params
        assert 0 <= self.opts.max_BUNs, (
            f"ERROR: Invalid max BUNs {self.opts.max_BUNs} provided. --max_BUNs cannot be negative."
        )
        assert 0 <= self.opts.max_BUPHs, (
            f"ERROR: Invalid max BUPHs {self.opts.max_BUPHs} provided. --max_BUPHs cannot be negative."
        )
        assert 0.0 <= self.opts.min_sat <= 2.0, (
            f"ERROR: Invalid min Saturation {self.opts.min_sat} provided. --min_sat must be in the range [0.0, 2.0]."
        )
        assert 0 <= self.opts.min_core_res, (
            f"ERROR: Invalid min core residues {self.opts.min_core_res} provided. --min_core_res must be a non-negative integer."
        )

        # Parsing params - can't validate until PDB is loaded
        if self.opts.fixed_res is not None:
            n_fixed_res = len(self.opts.fixed_res.split(","))
            assert (self.opts.n_res - n_fixed_res) > 0, f"Network size ({self.opts.n_res}) must be larger than number of fixed residues ({n_fixed_res})"
            assert n_fixed_res > 0, "You must provide at least one fixed residue if --fixed_res is specified."

    def run(self):
        """
        Run the HBDesigner inference pipeline with the validated options.
        """
        t0 = time.time()
        # 1. Parse input file with specified args
        print(f"Running HBDesigner for input {self.opts.pdb}...")
        try:
            self.scaffold = Protein.from_pdb_file(self.opts.pdb)

            if self.opts.fixed_res is not None:
                fixed_res = validate_residues(self.scaffold, self.opts.fixed_res, mode="fixed")
            else:
                fixed_res = np.empty((0,), dtype=np.int64)

            scaffold_copy = deepcopy(self.scaffold)
            self.scaffold.clear_sequence()

            # Impute fixed residues back in
            self.scaffold.aatype[fixed_res] = scaffold_copy.aatype[fixed_res]
            self.scaffold.atom27_xyz[fixed_res, ...] = scaffold_copy.atom27_xyz[fixed_res, ...]
            self.scaffold.atom27_mask[fixed_res, ...] = scaffold_copy.atom27_mask[fixed_res, ...]

            if self.opts.sel_chains is not None:
                self.scaffold, scaffold_unused = extract_chains(
                    self.scaffold, self.opts.sel_chains
                )
            symmetry_mask = get_symmetry_mask(self.scaffold, self.opts.symm_chains)

            if self.opts.guide_res is not None:
                guide_res = validate_residues(self.scaffold, self.opts.guide_res, mode="guide")
                guide_res_xyz = self.scaffold.atom27_xyz[guide_res, :4, :]
            else:
                guide_res = None
                guide_res_xyz = None

        except Exception:
            raise ValueError(
                f"ERROR: HBDesigner failed to parse PDB file {self.opts.pdb}!"
            )

        # 2. Initialize models
        print(f"Loading design model from checkpoint {self.opts.design_ckpt}...")
        design_model = self.load_design_model(guide_res)

        print(f"Loading packing model for packer {self.opts.packer}...")
        initialize_rosetta()
        packing_model = self.load_packing_model()

        print(
            f"TIME: {(time.time() - t0):.3f}s for input parsing and model initialization."
        )

        # 3. Generate design sequences
        ttime = time.time()
        samples = self.sample_from_hbdesigner(design_model, guide_res, fixed_res)
        n_samples = len(samples)
        print(
            f"Finished generating {n_samples} unique samples with HBDesigner in {time.time() - ttime:.3f} sec"
        )

        # 4. Pack and score sequences
        ttime = time.time()
        results = self.pack_samples(samples=samples, model=packing_model)
        print(
            f"Finished packing/scoring {n_samples} samples with {self.opts.packer} in {time.time() - ttime:.3f} sec"
        )

        # 5. Rank and save outputs
        if len(results) < 1:
            print("No valid networks passed score filters. Ending run.")
            return
        ttime = time.time()
        if self.opts.symm_chains is None:
            symmetry_mask = None

        # Put unused chains back in with proper order
        if self.opts.sel_chains is not None:
            for key, value in results.items():
                results[key]["protein"] = concat_proteins(
                    [value["protein"], scaffold_unused]
                )

        if self.opts.packer != "none":
            self.rank_and_save(
                results=results,
                symmetry_mask=symmetry_mask,
                guide_res_xyz=guide_res_xyz,
            )
        else:
            self.rank_and_save_none(
                results=results,
                symmetry_mask=symmetry_mask,
                guide_res_xyz=guide_res_xyz,
            )
        print(
            f"Finished ranking and saving {self.opts.top_k} samples in {time.time() - ttime:.3f} sec"
        )

        print(
            f"Total runtime for {self.opts.n_samples} samples of {self.opts.n_res} res on protein with {self.scaffold.n_res} res: {time.time() - t0:.3f} sec"
        )
        return

    def rank_and_save(
        self,
        results: Dict[str, Any],
        guide_res_xyz: np.ndarray = None,
        symmetry_mask: np.ndarray = None,
    ) -> None:
        df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)
        df.sort_values(
            by=[
                "buried_heavy_unsats",
                "buried_unsat_Hpol",
                "saturation",
                "HB_Score_full",
            ],
            ascending=[True, True, False, True],
            inplace=True,
        )

        if guide_res_xyz is not None:
            guide_atom_exact = get_guide_atom(guide_res_xyz, sigma=0.01)

        print("Ranking and saving outputs...\n", "=" * 100)
        rows = np.arange(df.shape[0])

        # Symmetrization is slow, so we parallelize it if we can
        if symmetry_mask is not None:
            with Pool(self.opts.n_workers) as p:
                result = p.map(
                    partial(
                        symmetrize_output,
                        df=df,
                        symm_mask=symmetry_mask,
                        clash_thresh=3.0,
                    ),
                    rows,
                )
                for i, r in enumerate(result):
                    df.iloc[i] = r
            # Drop identical symmetrized networks (keep best-scoring one)
            df = df.drop_duplicates(subset=["network"], keep="first")
            df = df.dropna()

        # Do top-K selection AFTER symmetrization to avoid under-sampling
        top_k = min(self.opts.top_k, df.shape[0])
        df_top = df.iloc[:top_k]

        for i in range(top_k):
            net = df_top.iloc[i]
            print(
                f"Rank {i + 1} \t(Sample {net.sample_num + 1}): \tBUHs: {net.buried_heavy_unsats}, \tBUPHs: {net.buried_unsat_Hpol}, \tSat: {net.saturation:.3f}, \tHB_Score: {net.HB_Score_full:.3f}, \tHB_Score_hb: {net.HB_Score_hb:.3f}, \tNetwork: {net.network}"
            )
            if self.opts.out_dir is not None:
                prefix = os.path.basename(self.opts.pdb).removesuffix(".pdb")
                fpath = os.path.join(
                    self.opts.out_dir, f"{prefix}_HBDes_rank_{i + 1}.pdb"
                )
                with open(fpath, "w") as fopen:
                    if guide_res_xyz is not None:
                        net.protein = add_guide_atom(net.protein, guide_atom_exact)
                    fopen.write(net.protein.to_pdb(unk_to_gly=True, no_hetatm=False))

        print("-" * 50)
        df_top = df_top.reset_index(drop=True)
        if self.opts.out_dir is not None:
            df_top["Rank"] = df_top.index
            df_top.to_csv(
                os.path.join(self.opts.out_dir, f"{prefix}_HBDes_stats.csv"),
                index=True,
                columns=[
                    "Rank",
                    "HB_Score_full",
                    "HB_Score_hb",
                    "Avg_Burial",
                    "saturation",
                    "buried_heavy_unsats",
                    "buried_unsat_Hpol",
                    "network",
                ],
            )
        return

    def rank_and_save_none(
        self,
        results: Dict[str, Any],
        guide_res_xyz: np.ndarray = None,
        symmetry_mask: np.ndarray = None,
    ) -> None:
        """Alternate ranking scheme using sequence model confidence."""
        df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)
        df.sort_values(
            by=["total_conf", "pos_conf", "seq_conf"],
            ascending=[False, False, False],
            inplace=True,
        )

        if guide_res_xyz is not None:
            guide_atom_exact = get_guide_atom(guide_res_xyz, sigma=0.01)

        print("Ranking and saving outputs...\n", "=" * 100)
        rows = np.arange(df.shape[0])

        # Symmetrization is slow, so we parallelize it if we can
        if symmetry_mask is not None:
            with Pool(self.opts.n_workers) as p:
                result = p.map(
                    partial(
                        symmetrize_output,
                        df=df,
                        symm_mask=symmetry_mask,
                        clash_thresh=3.0,
                    ),
                    rows,
                )
                for i, r in enumerate(result):
                    df.iloc[i] = r
            # Drop identical symmetrized networks (keep best-scoring one)
            df = df.drop_duplicates(subset=["network"], keep="first")

        # Do top-K selection AFTER symmetrization to avoid under-sampling
        top_k = min(self.opts.top_k, df.shape[0])
        df_top = df.iloc[:top_k]

        for i in range(top_k):
            net = df_top.iloc[i]
            print(
                f"Rank {i + 1} \t(Sample {net.sample_num + 1}): \tTotal Conf: {net.total_conf:.3f}, \tPos Conf: {net.pos_conf:.3f}, \tSeq Conf: {net.seq_conf:.3f} \tNetwork: {net.network}"
            )
            if self.opts.out_dir is not None:
                prefix = os.path.basename(self.opts.pdb).removesuffix(".pdb")
                fpath = os.path.join(
                    self.opts.out_dir, f"{prefix}_HBDes_rank_{i + 1}.pdb"
                )
                with open(fpath, "w") as fopen:
                    if guide_res_xyz is not None:
                        net.protein = add_guide_atom(net.protein, guide_atom_exact)
                    fopen.write(net.protein.to_pdb(unk_to_gly=True, no_hetatm=False))

        print("-" * 50)
        df_top = df_top.reset_index(drop=True)
        if self.opts.out_dir is not None:
            df_top["Rank"] = df_top.index
            df_top.to_csv(
                os.path.join(self.opts.out_dir, f"{prefix}_HBDes_stats.csv"),
                index=True,
                columns=[
                    "Rank",
                    "pos_conf",
                    "seq_conf",
                    "total_conf",
                    "network",
                ],
            )
        return

    def pack_samples(
        self,
        samples: List[Dict[str, List[int]]],
        model: Union[torch.nn.Module, None],
    ):
        """
        Route the samples to the correct packing method and return results.

        Arguments:
            samples (List[Dict[str, List[int]]]): List of samples to pack.
            model (torch.nn.Module): Packing model to use for packing the samples.
        """
        # Cropping during packing will mess up core mask, so we must get it beforehand
        core_mask = get_core_mask(deepcopy(self.scaffold), core_cutoff=5.2)
        # Need to set this after pyrosetta init but before packing/scoring
        set_real_option("score:hb_max_energy", self.opts.max_hb_energy)

        if self.opts.packer == "rosetta":
            results = self.pack_with_rosetta(
                samples=samples,
                core_mask=core_mask,
            )
        elif self.opts.packer == "hbpacker":
            results = self.pack_with_hbpacker(
                samples=samples,
                core_mask=core_mask,
                model=model,
            )
        elif self.opts.packer == "pippack":
            results = self.pack_with_pippack(
                samples=samples,
                core_mask=core_mask,
                model=model,
            )
        elif self.opts.packer == "none":
            results = self.pack_with_none(
                samples=samples,
                core_mask=core_mask,
            )
        else:
            raise ValueError(
                f"Invalid packer {self.opts.packer} specified. Valid options are: 'rosetta', 'hbpacker', 'pippack', or 'none'."
            )
        return results

    def pack_with_none(
        self,
        samples: List[Dict[str, Any]],
        core_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Instead of packing, just initialize random sidechains for output."""
        results = {}
        total_chains = np.unique(self.scaffold.chain_index).size

        for i, sample in enumerate(samples):
            # Apply sampled aatypes to scaffold
            p = deepcopy(self.scaffold)
            p.aatype[:] = rc.restype_order["G"]
            p.aatype[sample["net_res"]] = sample["seq"]

            # Initialize sidechains with zeroed chi angles
            chi_angles = np.zeros((p.n_res, 4))
            chi_mask = [rc.chi_angles_mask[aa] for aa in p.aatype]
            chi_mask = np.stack(chi_mask)  # [L, 4]

            atom14_xyz, atom14_mask = build_sc_from_chi(
                p.atom27_xyz[..., :4, :], p.aatype, chi_angles, chi_mask
            )
            p.atom27_xyz[..., 4:14, :] = atom14_xyz[..., 4:14, :]
            p.atom27_mask[..., 4:14] = atom14_mask[..., 4:14]

            # Collect chain and core res stats for filtering
            n_chains = np.unique(p.chain_index[sample["net_res"]]).size
            n_core_res = np.sum(core_mask[sample["net_res"]])

            if (n_core_res >= self.opts.min_core_res) and (n_chains == total_chains):
                results[hash(p.to_pdb(unk_to_gly=True))] = {
                    "network": get_network_res(p),
                    "protein": p,
                    "pos_conf": np.mean(sample["net_res_probs"]),
                    "seq_conf": np.mean(sample["seq_probs"]),
                    "total_conf": np.mean(sample["seq_probs"])
                    * np.mean(sample["net_res_probs"]),
                    "sample_num": i,
                    "n_chains": n_chains,
                    "n_core_res": n_core_res,
                }

        return results

    @torch.no_grad()
    def pack_with_hbpacker(
        self,
        samples: List[Dict[str, Any]],
        model: HBDesigner,
        core_mask: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Use HBPacker to pack a list of predictions.

        Arguments:
            samples (List[Dict[str, Any]]): List of samples to pack.
            model (HBDesigner): Loaded HBDesigner model for packing.
            core_mask (np.ndarray): Core mask for the scaffold.

        Returns:
            Dict[str, Any]: Packed results with scores.
        """
        # Loop over batches of predictions
        all_packs = []
        batch_size = 10_000
        n_designs = len(samples)

        t0 = time.time()
        # Configure DataLoader to take advantage of multiprocessing
        ds = InferenceDataset(
            deepcopy(self.scaffold), samples, batch_size, self.opts.pack_crop
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=None,
            num_workers=self.opts.n_workers,
            collate_fn=HBPackerDataset.collate,
            persistent_workers=True,
        )
        dl = iter(dl)

        # Iterate over dataloader with multiproc enabled
        while True:
            batch = next(dl)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                packs = model.run_pack_recyc(
                    batch.to("cuda" if torch.cuda.is_available() else "cpu"),
                    n_recycles=model.cfg.model.hbpacker.num_recycles,
                )
            packs = packs.to("cpu")
            all_packs.extend(packs.to_data_list())
            if len(all_packs) == len(ds):
                break

        del dl
        del ds
        t1 = time.time()
        print(f"Packed {n_designs} designs with HBPacker in {(t1 - t0):.3f} sec")

        print("Running minimization and scoring on packed designs...")
        with Pool(self.opts.n_workers) as p:
            results = p.map(
                partial(
                    minimize_and_score_network,
                    scaffold=deepcopy(self.scaffold),
                    minimize=True,
                    core_mask=core_mask,
                ),
                all_packs,
            )
        t2 = time.time()
        print(f"Scored {n_designs} designs with Rosetta in {(t2 - t1):.3f} sec")

        results = self.filter_packs(results)
        print(f"Successes: {len(results)} / {len(all_packs)}")
        return results

    def pack_with_rosetta(
        self,
        samples: List[Dict[str, List[int]]],
        core_mask: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Pack a list of samples onto a scaffold with Rosetta, as fast as possible.

        Arguments:
            samples (List[Dict[str, Any]]): List of samples to pack.
            model (HBDesigner): Loaded HBDesigner model for packing.
            core_mask (np.ndarray): Core mask for the scaffold.

        Returns:
            Dict[str, Any]: Packed results with scores.
        """

        # Pack and score samples in parallel
        with Pool(self.opts.n_workers) as p:
            results = p.map(
                partial(
                    pack_and_score_network,
                    scaffold=deepcopy(self.scaffold),
                    minimize=True,
                    pack_crop=self.opts.pack_crop,
                    core_mask=core_mask,
                    max_BUNs=self.opts.max_BUNs,
                ),
                samples,
            )

        results = self.filter_packs(results)

        print(f"Successes: {len(results)} / {len(samples)}")
        return results

    @torch.no_grad()
    def pack_with_pippack(
        self,
        samples: List[Dict[str, Any]],
        model: List[PIPPackFineTune],
        core_mask: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Use PIPPack to pack a list of predictions.

        Arguments:
            samples (List[Dict[str, Any]]): List of samples to pack.
            model (List[PIPPackFineTune]): Loaded PIPPack model for packing.
            core_mask (np.ndarray): Core mask for the scaffold.

        Returns:
            Dict[str, Any]: Packed results with scores.
        """
        # Loop over batches of predictions
        all_packs = []
        batch_size = 10_000
        n_designs = len(samples)

        t0 = time.time()
        # Configure DataLoader to take advantage of multiprocessing
        ds = InferenceDataset(
            deepcopy(self.scaffold), samples, batch_size, self.opts.pack_crop
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=None,
            num_workers=self.opts.n_workers,
            collate_fn=HBPackerDataset.collate,
            persistent_workers=True,
        )
        dl = iter(dl)
        dev = next(model[0].parameters()).device

        # Iterate over dataloader with multiproc enabled
        while True:
            batch = next(dl)

            # Need to reformat for PIPPack
            batch["aatype"][batch["aatype"] == rc.restype_num] = rc.restype_order["A"]
            batch["atom14_xyz"][:, 4] = impute_CB(
                batch["atom14_xyz"][:, 0],
                batch["atom14_xyz"][:, 1],
                batch["atom14_xyz"][:, 2],
            )
            batch["atom14_mask"][:, 4] = torch.prod(batch["atom14_mask"][:, :4], dim=-1)
            batch["sc_dihedral"], batch["sc_dihedral_mask"] = calc_sc_dihedrals(
                batch["atom14_xyz"], batch["aatype"]
            )
            batch["sc_dihedral"] = torch.zeros_like(batch["sc_dihedral"])
            batch["sc_dihedral_mask"] = torch.zeros_like(batch["sc_dihedral_mask"])

            # Get and apply PIPPack preds
            all_logits = []
            for m in model:
                logits = (
                    get_sidechain_logits(
                        m,
                        batch.to(dev),
                        recycles=1,
                    )
                    .detach()
                    .cpu()
                )
                all_logits.append(logits)

            # Stack and avg logits from ensemble
            logits = torch.mean(torch.stack(all_logits, dim=-1), dim=-1)
            proteins, b_list = batch_to_proteins(batch)

            proteins = apply_logits_to_proteins(
                proteins,
                logits,
                resample=False,
            )
            # # Attach pack knn to proteins
            for p, b in zip(proteins, b_list):
                p.pack_knn = b.pack_knn

            all_packs.extend(proteins)
            if len(all_packs) == len(ds):
                break

        del dl
        del ds
        t1 = time.time()
        print(f"Packed {n_designs} designs with PIPPack in {(t1 - t0):.3f} sec")

        print("Running minimization and scoring on packed designs...")
        # Score samples in parallel
        with Pool(self.opts.n_workers) as p:
            results = p.map(
                partial(
                    minimize_and_score_network,
                    scaffold=deepcopy(self.scaffold),
                    minimize=True,
                    core_mask=core_mask,
                    max_BUNs=self.opts.max_BUNs,
                ),
                all_packs,
            )
        t2 = time.time()
        print(f"Scored {n_designs} designs with Rosetta in {(t2 - t1):.3f} sec")

        results = self.filter_packs(results)
        print(f"Successes: {len(results)} / {len(all_packs)}")
        return results

    def filter_packs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filter a list of packs based on user-defined criteria.

        Arguments:
            results (List[Dict[str, Any]]): List of packed samples with scores.

        Returns:
            Dict[str, Any]: Filtered results containing only valid packs.

        """

        total_chains = np.unique(self.scaffold.chain_index).size
        passed = {}
        for i, r in enumerate(results):
            if r is not None:
                if (
                    r["buried_heavy_unsats"] <= self.opts.max_BUNs
                    and r["n_chains"] == total_chains
                    and r["buried_unsat_Hpol"] <= self.opts.max_BUPHs
                    and r["saturation"] >= self.opts.min_sat
                    and r["n_core_res"] >= self.opts.min_core_res
                ):
                    passed[r["hash"]] = r
                    passed[r["hash"]]["sample_num"] = i

        return passed

    @torch.no_grad()
    def sample_from_hbdesigner(
        self,
        model: HBDesigner,
        guide_res: np.ndarray = None,
        fixed_res: np.ndarray = None,
    ) -> List[Dict[str, List[int]]]:
        """
        Generate n_samples samples from HBDesigner model.

        Arguments:
            model (HBDesigner): Loaded HBDesigner model.
            guide_res (np.ndarray, optional): Guide residues for triangulating virtual guide atom.
            fixed_res (np.ndarray, optional): Fixed residues that are already present in the network.
        Returns:
            List[Dict[str, List[int]]]: List of unique predictions from the model.

        """
        # Only need to featurize once
        data = HBDesignerDataset.featurize_inference(
            deepcopy(self.scaffold),
            n_res=self.opts.n_res,
            guide_res=guide_res,
            guide_radius=self.opts.guide_radius,
            guide_seq=self.opts.guide_seq,
            min_burial=self.opts.min_burial,
            fixed_res=fixed_res,
        )

        # Check there are enough designable positions
        n_des = data.des_mask.sum()
        n_tot = data.des_mask.numel()
        print(f"Total Positions: {n_tot} \nDesignable Positions: {n_des}")
        assert n_des >= self.opts.n_res, (
            f"ERROR: Number of designable positions ({n_des}) is less than desired network size ({self.opts.n_res})! Try adjusting your --guide_radius or --min_burial criteria."
        )

        # Sampling params
        batch_size = 10_000
        num_res = data.num_nodes
        # Generate N unique preds
        unique_preds, unique_preds_full = [], []
        dev = next(model.parameters()).device

        res_temp_inc_per_invalid = 1e-3  # to avoid infinite loop
        res_sample_temp = self.opts.T_range[0]
        seq_sample_temp = self.opts.T_range[0]

        temp_inc = 0.0
        _last_print = 0
        batches = 0
        n_copies = batch_size // num_res
        MAX_SAMPLES = 100 * n_copies

        # Do collation and transfer ONCE and re-use batch
        batch = HBDesignerDataset.collate([data] * n_copies)
        batch.net_res_num[:] = self.opts.n_res - fixed_res.size
        batch.to(dev)

        while len(unique_preds) < self.opts.n_samples:
            # Print progress
            res_sample_temp_c = min(
                res_sample_temp + (temp_inc / 2.0), self.opts.T_range[1]
            )
            seq_sample_temp_c = min(seq_sample_temp + temp_inc, self.opts.T_range[1])
            if len(unique_preds) > _last_print + 100:
                _last_print = len(unique_preds)
                print(
                    f"Unique predictions: {len(unique_preds)}/{self.opts.n_samples} in {batches} batches so far..."
                )
                print(
                    f"Current temps (res/seq): {res_sample_temp_c:.3f}/{seq_sample_temp_c:.3f}"
                )
            # Mixed precision inference is ~50% faster w/o dropping any performance
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = model.sample_new(
                    batch.clone(),
                    res_sample_temp=res_sample_temp_c,
                    seq_sample_temp=seq_sample_temp_c,
                    bb_noise=self.opts.bb_noise,
                )

            # Convert predictions to lists
            results = [
                {k: v.cpu().tolist() for k, v in results[r].items()} for r in results
            ]

            # Collect unique predictions
            for r in results:
                unique_r = dict((k, r[k]) for k in ["seq", "net_res"])
                if unique_r not in unique_preds:
                    unique_preds.append(unique_r)
                    unique_preds_full.append(r)
                else:
                    temp_inc += res_temp_inc_per_invalid
            batches += 1

            if (batches * n_copies) > MAX_SAMPLES:
                print(
                    f"Warning: max sample count {MAX_SAMPLES} reached, ending sampling with only {len(unique_preds)} unique samples."
                )
                break
        print(f"Batches: {batches}\tSamples: {batches * (batch_size // num_res)}")
        return unique_preds_full

    def load_design_model(self, guide_res: str = None) -> HBDesigner:
        """
        Set up design config options and load model weights.

        Arguments:
            guide_res (str, optional): Guide residues for triangulating virtual guide atom.

        Returns:
            HBDesigner: Loaded HBDesigner design model.
        """
        self.design_cfg.log_dir = None
        self.design_cfg.num_workers = 0
        self.design_cfg.model.hbdesigner.guide_atom_pct = (
            0.0 if guide_res is None else 1.0
        )
        self.design_cfg.model.hbdesigner.guide_atom_sigma = 4.0
        return load_HBDesigner(
            self.design_cfg, self.opts.design_ckpt, self.design_cfg.device
        )

    def load_packing_model(
        self,
    ) -> torch.nn.Module:
        """
        Set up packing config options and load model weights, if requested.
        """
        self.pack_cfg.log_dir = None
        self.pack_cfg.num_workers = 0

        if self.opts.packer == "hbpacker":
            print("Using HBPacker for HBDesigner inference...")

            # HBPacker params
            self.pack_cfg.model.hbpacker.pack_method = "hbpacker"
            self.pack_cfg.model.hbpacker.pack_mode = "fast"
            self.pack_cfg.model.hbpacker.bb_noise = 0.0
            packer = load_HBPacker(
                self.pack_cfg, self.opts.pack_ckpt, self.pack_cfg.device
            )
            return packer

        elif self.opts.packer == "pippack":
            print("Using PIPPack for HBDesigner inference...")

            # PIPPack params
            self.pack_cfg.model.pippack.recycles = 1
            n_models = 3
            pippack_ckpt = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.pack_cfg.model.pippack.ckpt = os.path.join(
                pippack_ckpt, "model/weights/pippack_model_1_ckpt.pt"
            )

            pippack = []
            models = ["1", "2", "3"][:n_models]
            for m in models:
                ckpt = self.pack_cfg.model.pippack.ckpt
                ckpt = "_".join(ckpt.split("_")[:-2]) + f"_{m}_ckpt.pt"
                self.pack_cfg.model.pippack.ckpt = ckpt
                mod = load_PIPPack(self.pack_cfg.model)
                pippack.append(mod)
            return pippack

        elif self.opts.packer == "rosetta":
            print("Using Rosetta packer for HBDesigner inference...")
            return None

        elif self.opts.packer == "none":
            print("Skipping packing step entirely...")
            return None
        else:
            raise ValueError(
                "Invalid packer specified. Options are 'hbpacker', 'rosetta', 'pippack', or 'none'."
            )


if __name__ == "__main__":
    parser = get_hbdes_parser()
    args = parser.parse_args(sys.argv[1:])

    model_runner = HBDesRunner(args)
    model_runner.run()
