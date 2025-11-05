import argparse
import datetime
import itertools
import os
import random
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch_geometric.data as gd
import wandb
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist

from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.core.select.util import calc_sc_neighbors
from pyrosetta import Pose

import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import (
    calc_bb_dihedrals,
    impute_CB,
    build_sc_from_chi,
)
from hbdesigner.data.hbnet import (
    get_guide_atom,
    get_seq_cond,
    calc_seq_rec_batched,
    crop_by_distance,
    score_protein,
)

from hbdesigner.data.protein import PDB_CHAIN_IDS, Protein
from hbdesigner.scripts.preprocess_asmbs import load_metadata
from hbdesigner.train.config import TrainConfig
from hbdesigner.train.trainer import SupervisedTrainer
from hbdesigner.utils import cycle, seed_everything, worker_init, init_empty


class HBDesignerDataset(torch.utils.data.IterableDataset):
    """
    Dataset for HBDesigner3 model.

    Iterates over clusters and samples a random assembly, then collects any valid networks.
    Randomly samples from candidate networks according to config params.

    Expected dataset format:
        - ABCD.pt files for all PDBs
        - ABCD_N.npz files for all assemblies
        - ABCD_N.gml files for all assemblies
        - ABCD_N_hbnet.npz files for all HBNet designs

    """

    def __init__(self, cfg: TrainConfig, split: str = "train"):
        # Load metadata
        self.cfg = cfg
        self.split = split

        if self.cfg.model.model_name == "HBDesigner":
            self.model_cfg = self.cfg.model.hbdesigner
        elif self.cfg.model.model_name == "HBPacker":
            self.model_cfg = self.cfg.model.hbpacker
        else:
            raise ValueError(f"Unknown model name: {self.cfg.model.model_name}")

        self.dir = self.model_cfg.data_location
        self.batch_size = self.model_cfg.batch_size
        self.MAX_LENGTH = self.batch_size
        df = load_metadata(self.dir, split=split)

        # Compile cluster -> chainid mapping
        self.cdict = {}
        for cluster in df["CLUSTER"].unique():
            self.cdict[cluster] = df.loc[df["CLUSTER"] == cluster]["CHAINID"].values
        self.keys = sorted(list(self.cdict.keys()))
        if self.split != "test":
            random.shuffle(self.keys)
        self.start, self.end = 0, len(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

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
        Yields batches of samples from a worker on infinite loop.
        Called by self.__iter__().

        Arguments:
            start (int): Start index of this worker's chunk of the dataset.
            end (int): End index of this worker's chunk of the dataset.

        Returns:
            List(gd.Data): List of Data objects representing a batch before collation.

        """
        while True:
            n_nodes = 0
            samples = []
            for idx in itertools.cycle(range(start, end)):
                # Build up batch to reach specified size
                sample = self.get(self.keys[idx])
                if sample is not None:
                    if n_nodes + sample.num_nodes > self.batch_size:
                        yield samples
                        n_nodes = 0
                        samples = []
                    n_nodes += sample.num_nodes
                    samples.append(sample)

    def get(self, idx: int) -> Union[gd.Data, None]:
        """
        Retrieves and featurizes a single query, if it is valid.
        Called by self.generate(idx), which is called by self.__iter__().

        Arguments:
            idx (int): Cluster index for sampling.

        Returns:
            gd.Data: torch_geometric Data object representing a featurized protein.
            OR
            None: if assembly is missing or otherwise invalid, None is returned (idx is skipped).

        """
        # Load a random assembly within the chosen cluster
        members = self.cdict[idx]
        # Deterministic sampling for test set
        if self.split != "test":
            pdb_ch = members[torch.randperm(len(members))[0]]
        else:
            pdb_ch = members[self.cfg.seed % len(members)]
        asmb = self._load_assembly(pdb_ch)
        # Skip edge cases
        if asmb is None:
            return None
        p, hbnet_arr = asmb

        if p.n_res > self.MAX_LENGTH:
            return None
        return self.featurize(p, hbnet_arr)

    def _sample_native_net(
        self, p: Protein, g: nx.DiGraph
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """
        Given a Protein and Graph of networks, return one valid network.

        Arguments:
            p (Protein): Protein to be sampled from
            g (nx.DiGraph): Graph representing network connectivity.

        Returns:
            np.ndarray: Array of network residue positions.
            np.ndarray: Copy of p.aatype with all residues except network set to GLY.
            OR
            None: Returned if no valid network exists in the given Graph.

        """
        cc = list(nx.weakly_connected_components(g))
        pos = []

        # Check each network for intra/inter-chain character
        sampling_probs = []
        for c in cc:
            sg = g.subgraph(c)
            n_res = len(sg)
            # If wrong size, don't sample
            if (n_res > self.model_cfg.max_res) or (
                n_res < self.model_cfg.min_res
            ):
                sampling_probs.append(0.0)
            else:
                nodes = list(sg.nodes(data=True))
                chains = list(set([n[-1]["PDB_Chain"] for n in nodes]))

                if self.model_cfg.rescore:
                    # Only filter if directed
                    if self.model_cfg.rescore_filter:
                        net_sat = nodes[0][-1]["Network_Sat"]
                        net_buns = nodes[0][-1]["Network_BUNs"]
                        net_buphs = nodes[0][-1]["Network_BUPHs"]
                        flag = (
                            (net_sat >= 0.5)
                            and (net_buns <= n_res // 4)
                            and (net_buphs <= 1 + (n_res // 4))
                        )
                    else:
                        flag = True
                    if flag:
                        sampling_probs.append(1.0)
                    else:
                        sampling_probs.append(0.0)
                else:
                    # Give different probs to inter/intra-chain cases
                    if len(chains) == 1:
                        sampling_probs.append(1.0)
                    else:
                        sampling_probs.append(self.model_cfg.inter_weight)

        # Check that at least one valid network exists
        if (len(sampling_probs) < 1) or (sum(sampling_probs) <= 0.0):
            return

        sampling_probs = np.array(sampling_probs) / sum(sampling_probs)
        # Deterministic sampling for test set
        if self.split != "test":
            c = np.random.choice(cc, size=1, p=sampling_probs)[0]
        else:
            cc = [c for i, c in enumerate(cc) if sampling_probs[i] > 0.0]
            if len(cc) < 1:
                return
            c = cc[self.cfg.seed % len(cc)]

        nodes = g.subgraph(c).nodes(data=True)
        for n in nodes:
            ch = PDB_CHAIN_IDS.index(n[-1]["PDB_Chain"])
            num = n[-1]["PDB_Number"]
            pdb_pos = np.where((p.residue_index == num) * (p.chain_index == ch))[0]
            pos.append(pdb_pos)

        hbnet_aatype = deepcopy(p.aatype)
        hbnet_aatype[:] = rc.restype_order["G"]
        hbnet_pos = np.array(pos)
        hbnet_aatype[hbnet_pos] = p.aatype[hbnet_pos]
        return hbnet_pos, hbnet_aatype

    def _load_assembly(self, pdb_ch: str) -> Union[Tuple[Protein, np.ndarray], None]:
        """
        Load assembly data, if present, for a PDB chain.

        Arguments:
            pdb_ch (str): Query string formatted in pdb_chain format (e.g., 1ABC_A).

        Returns:
            Protein: Loaded Protein object.
            np.ndarray: HB network residue aatype array.
            OR
            None: If any files are missing from the entry or assembly, returns None (entry is skipped).
        """
        pdbid, chid = pdb_ch.split("_")
        PREFIX = os.path.join(self.dir, pdbid[1:3], pdbid)
        # Check if whole assembly is missing
        pdb_pt = PREFIX + ".pt"
        if not os.path.isfile(pdb_pt):
            return

        # Find candidate assemblies which contain chid chain
        meta = torch.load(pdb_pt, weights_only=False)
        asmb_ids = meta["asmb_ids"]
        asmb_chains = meta["asmb_chains"]
        asmb_candidates = set(
            [a for a, b in zip(asmb_ids, asmb_chains) if chid in b.split(",")]
        )

        # Pick random assembly
        asmb_candidates = sorted(list(asmb_candidates))
        # If chain is missing from all assemblies, skip it
        if len(asmb_candidates) < 1:
            return
        if self.split != "test":
            asmb_id = asmb_candidates[torch.randperm(len(asmb_candidates))[0]]
        else:
            asmb_id = asmb_candidates[self.cfg.seed % len(asmb_candidates)]
        asmb_npz = PREFIX + f"_{asmb_id}.npz"

        if self.model_cfg.rescore:
            asmb_gml = PREFIX + f"_{asmb_id}_rescore.gml"
        else:
            asmb_gml = PREFIX + f"_{asmb_id}.gml"

        # Check if files exist
        if not os.path.isfile(asmb_npz) or not os.path.isfile(asmb_gml):
            return

        # Load Protein and Graph data
        p = np.load(asmb_npz)
        g = nx.read_gml(asmb_gml)

        p = Protein(
            atom27_xyz=p["atom27_xyz"],
            atom27_mask=p["atom27_mask"],
            aatype=p["aatype"],
            residue_index=p["residue_index"],
            chain_index=p["chain_index"],
            b_factors=p["b_factors"],
        )

        # Native network sampling
        network = self._sample_native_net(p, g)
        if network is None:
            return
        else:
            hbnet_pos, hbnet_arr = network

        # Option to use HBNet designs (data augmentation)
        asmb_hbnet = asmb_gml.removesuffix(".gml") + "_hbnet.npz"
        if os.path.isfile(asmb_hbnet):
            use_hbnet = np.random.binomial(
                n=1, p=self.model_cfg.hbnet_pct, size=1
            ).astype(bool)[0]
            if use_hbnet:
                # Grab a random network
                npy = np.load(asmb_hbnet)
                keys = list(npy.keys())
                assert len(keys) % 3 == 0
                num_designs = len(keys) // 3
                net = np.random.choice(np.arange(num_designs))

                # Overwrite hbnet arrays
                hbnet_aatype, hbnet_pos = npy[f"net{net}_aatype"], npy[f"net{net}_pos"]
                hbnet_arr = deepcopy(p.aatype)
                hbnet_arr[:] = rc.restype_order["G"]
                hbnet_arr[hbnet_pos] = hbnet_aatype

                # Update xyz coords with hbnet coords for packer
                hbnet_xyz = npy[f"net{net}_atom14_xyz"]
                p.clear_sequence()
                p.atom27_xyz[hbnet_pos, :14, :] = hbnet_xyz
                p.atom27_mask[hbnet_pos, :14] = np.prod(
                    p.atom27_xyz[hbnet_pos, :14, :] != 0, axis=-1
                )
                p.aatype[hbnet_pos] = hbnet_arr[hbnet_pos]

        # If HBNet Pct is 100%, don't use any native nets
        elif self.model_cfg.hbnet_pct >= 1.0:
            return

        # Prune chains not included in network
        network_chains = np.unique(p.chain_index[hbnet_pos])
        chain_mask = (
            np.sum((p.chain_index[:, None] == network_chains[None, :]), axis=-1) > 0
        )
        p = p.mask(np.where(chain_mask)[0])
        hbnet_arr = hbnet_arr[chain_mask]
        return p, hbnet_arr

    def featurize(self, p: Protein, hbnet_arr: np.ndarray) -> gd.Data:
        """
        Featurize Protein and Graph info into HBDesigner model inputs.

        Arguments:
            p (Protein): Protein object.
            hbnet_arr (np.ndarray): Copy of p.aatype with all non network residues set to GLY.

        Returns:
            gd.Data: torch_geometric Data object with featurized protein.
        """

        c = self.cfg.model.hbdesigner
        # Override seq for HBNet
        hbnet_pos = np.where(hbnet_arr != rc.restype_order["G"])[0]

        if (hbnet_pos.size > c.max_res) or (hbnet_pos.size < c.min_res):
            return

        # Check for invalid restypes in network res
        invalid = hbnet_arr[hbnet_pos][:, None] == rc.restype_non_hb_idx[None, :]
        if np.sum(invalid) > 0.0:
            return

        # Do step and residue sampling for training
        n_res = hbnet_pos.size

        # Always run test from tstep 0 to get accurate cond info
        if self.split == "test":
            tstep = 0
        else:
            tstep = random.randint(0, n_res - 1)

        # Randomly grab res done from options
        res_done = np.random.choice(hbnet_pos, size=tstep, replace=False)
        # Turn on loss mask for remaining positions
        nll_mask = np.zeros_like(p.aatype, dtype=np.int32)
        res_remain = np.setdiff1d(hbnet_pos, res_done)
        nll_mask[res_remain] = 1
        # Make mask for completed positions (used in loss calc)
        done_mask = np.zeros_like(p.aatype, np.int32)
        done_mask[res_done] = 1

        # Clear any non-network res for ground-truth aatype and xyz
        hbnet_mask = (nll_mask + done_mask) > 0
        p.aatype[~hbnet_mask] = rc.restype_num
        p.aatype[hbnet_pos] = hbnet_arr[hbnet_pos]
        aatype_gt = p.aatype
        p.atom27_xyz[~hbnet_mask, 4:] = 0.0
        p.atom27_mask[~hbnet_mask, 4:] = 0
        residue_index = p.residue_index
        chain_index = p.chain_index

        # End of ground-truth features
        p = deepcopy(p)
        # Clear any not-yet-decoded residues for input aatype
        p.aatype[nll_mask > 0] = rc.restype_num
        aatype = p.aatype
        # Clear ALL sidechains from input feats
        p.atom27_xyz[:, 4:] = 0.0
        p.atom27_mask[:, 4:] = 0.0
        atom14_xyz = p.atom27_xyz[:, :14]
        atom14_mask = p.atom27_mask[:, :14]

        # Get guide atom before bb xyz noising
        guide_atom_xyz = get_guide_atom(
            atom14_xyz[hbnet_pos, :3, :], c.guide_atom_sigma
        )

        # Noise bb xyz to avoid seq rec cheating
        if (self.split == "train") and (c.bb_noise > 0.0):
            atom14_xyz[..., :5, :] = atom14_xyz[..., :5, :] + (
                self.cfg.model.hbdesigner.bb_noise
                * np.random.randn(*atom14_xyz[..., :5, :].shape)
            )

        # Use noised xyz to get bb dihedrals
        bb_dihedral = calc_bb_dihedrals(atom14_xyz, p.residue_index, return_mask=False)

        # Create the Data object
        protein_data = gd.Data(
            num_nodes=aatype.shape[0],
            x=torch.zeros((1, 1)),  # x is used often to identify the device
            # Ground truth seq
            aatype_gt=torch.from_numpy(aatype_gt).to(torch.long),  # [L]
            # Input seq
            aatype=torch.from_numpy(aatype).to(torch.long),  # [L]
            atom14_xyz=torch.from_numpy(atom14_xyz).to(torch.float32),  # [L, 14, 3]
            atom14_mask=torch.from_numpy(atom14_mask).to(torch.float32),  # [L, 14]
            residue_index=torch.from_numpy(residue_index).to(torch.int32),  # [L]
            chain_index=torch.from_numpy(chain_index).to(torch.int32),  # [L]
            bb_dihedral=torch.from_numpy(bb_dihedral).to(torch.float32),  # [L, 3]
            # Masks and cond info
            nll_mask=torch.from_numpy(nll_mask).to(torch.float32),  # [L]
            done_mask=torch.from_numpy(done_mask).to(torch.long),  # [L]
            guide_atom_xyz=torch.from_numpy(guide_atom_xyz).to(torch.float32),  # [1, 3]
        )

        # If not using seq cond, set all res to UNK
        hbnet_res = aatype_gt[res_remain]
        unk_all = np.random.binomial(n=1, p=(1 - c.seq_cond_pct), size=1).astype(bool)[
            0
        ]
        if unk_all:
            hbnet_res[:] = rc.restype_num

        # If using seq cond, check if doing partial or full cond
        else:
            unk_some = np.random.binomial(n=1, p=c.seq_cond_unk_pct, size=1).astype(
                bool
            )[0]
            # If partial, set a random subset of residues to UNK
            # If full, don't change any residues
            if unk_some:
                sample_probs = {
                    1: [0.50, 0.50],
                    2: [0.00, 1.00, 0.00],
                    3: [0.00, 0.50, 0.50, 0.00],
                    4: [0.00, 0.25, 0.50, 0.25, 0.00],
                    5: [0.00, 0.16, 0.34, 0.34, 0.16, 0.00],
                    6: [0.00, 0.10, 0.24, 0.32, 0.24, 0.10, 0.00],
                }
                probs = sample_probs[hbnet_res.size]
                n_unk = np.random.choice(np.arange(len(probs)), 1, p=probs)
                flips = np.random.choice(
                    np.arange(hbnet_res.size), size=n_unk, replace=False
                )
                hbnet_res[flips] = rc.restype_num

        protein_data["aatype_cond"] = torch.from_numpy(get_seq_cond(hbnet_res)).to(
            torch.float32
        )  # [1, 21]

        protein_data["c_idx"] = protein_data["chain_index"]
        return protein_data

    @staticmethod
    def collate(data_list: Sequence[gd.Data]) -> gd.Batch:
        # follow_batch will create extra tensors for provided keys,
        # e.g. 'aatype' will add 'aatype_batch'

        if isinstance(data_list[0], list):
            data_list = [i for sub in data_list for i in sub]

        batch = gd.Batch.from_data_list(
            data_list,
            follow_batch=[
                "aatype",
            ],
        )
        # _index is auto incremented - need to fix this
        batch.chain_index = batch.c_idx
        batch.batch2res_repeats = torch.tensor(
            [(batch.aatype_batch == j).sum() for j in range(batch.num_graphs)],
        )  # [B,]

        # Get remaining res per set
        batch.net_res_num = torch.tensor(
            [
                (batch.nll_mask[batch.aatype_batch == j]).sum()
                for j in range(batch.num_graphs)
            ],
        ).long()  # [B,]
        return batch

    @staticmethod
    def featurize_inference(
        p: Protein,
        n_res: int = 2,
        guide_res: np.ndarray = None,
        guide_radius: float = 1e6,
        guide_seq: str = None,
        min_burial: float = 0.0,
        fixed_res: np.ndarray = None
    ) -> gd.Data:
        """
        Featurize Protein for HBDesigner inference. Unlike for training, we have no ground truth network here.

        Arguments:
            p (Protein): Protein object.
            n_res (int): Number of residues to include in predicted network. Default is 2.
            guide_res (np.ndarray): Array of positions for inclusion in guide atom centroid calculation. Default is None (ignored).
            guide_radius (float): Radius around the guide atom to allow designable. Default is 1e6 (all residues).
            guide_seq (str): Guide sequence to enforce in all designs. Default is None (all UNK).
            min_burial (float): Minimum burial value to allow designable. Default is 0.0. Core is 5.2, Surface is 2.0.
            fixed_res (np.ndarray): Array of positions that are already present in the network. Default is None (no fixed residues).

        Returns:
            gd.Data: torch_geometric Data object with featurized protein.
        """
        # Protein features
        p_copy = deepcopy(p)
        p.clear_sequence()

        # Impute aatype and xyz info back in for fixed residues
        p.aatype[fixed_res] = p_copy.aatype[fixed_res]
        p.atom27_xyz[fixed_res, :, :] = p_copy.atom27_xyz[fixed_res, :, :]
        p.atom27_mask[fixed_res, :] = p_copy.atom27_mask[fixed_res, :]
        # Adjust n_res down for fixed_res
        n_res -= fixed_res.size

        aatype = p.aatype
        atom14_xyz = p.atom27_xyz[:, :14]
        atom14_mask = p.atom27_mask[:, :14]
        residue_index = p.residue_index
        chain_index = p.chain_index
        bb_dihedral = calc_bb_dihedrals(
            p.atom27_xyz[:, :14], p.residue_index, return_mask=False
        )

        # Create the Data object
        protein_data = gd.Data(
            num_nodes=aatype.shape[0],
            x=torch.zeros((1, 1)),  # x is used often to identify the device
            aatype=torch.from_numpy(aatype).to(torch.long),  # [L]
            atom14_xyz=torch.from_numpy(atom14_xyz).to(torch.float32),  # [L, 14, 3]
            atom14_mask=torch.from_numpy(atom14_mask).to(torch.float32),  # [L, 14]
            residue_index=torch.from_numpy(residue_index).to(torch.int32),  # [L]
            chain_index=torch.from_numpy(chain_index).to(torch.int32),  # [L]
            bb_dihedral=torch.from_numpy(bb_dihedral).to(torch.float32),  # [L, 3]
        )

        # nll_mask is 1 for designable positions, 0 otherwise
        nll_mask = np.ones_like(p.aatype, dtype=np.int32)
        nll_mask[fixed_res] = 0
        protein_data["nll_mask"] = torch.from_numpy(nll_mask).to(torch.float32)
        protein_data["aatype_masked"] = torch.from_numpy(p.aatype).to(torch.long)

        # done_mask is 1 for already-designed positions, 0 otherwise
        done_mask = np.zeros_like(p.aatype, np.int32)
        done_mask[fixed_res] = 1
        protein_data["done_mask"] = torch.from_numpy(done_mask).to(torch.long)

        # Guide atom cond info
        guide_atom_sigma = 4.0  # sd of guide atom sampling distribution
        if guide_res is None:
            guide_atom_xyz = np.zeros_like(
                atom14_xyz[0, 0:1, :],
            )
            des_mask = np.prod(atom14_mask[..., :4], axis=-1)
        else:
            guide_atom_xyz = get_guide_atom(
                atom14_xyz[guide_res, :3, :], guide_atom_sigma
            )
            # Make mask of nearby residues
            cb_xyz = impute_CB(
                atom14_xyz[..., 0, :], atom14_xyz[..., 1, :], atom14_xyz[..., 2, :]
            )
            cb_dist = np.squeeze(cdist(cb_xyz, guide_atom_xyz))  # [L]
            des_mask = (cb_dist <= guide_radius) * np.prod(
                atom14_mask[..., :4], axis=-1
            )

        # Add burial constraint to designable positions
        pose = Pose()
        pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))
        sc_neighbors = np.array(calc_sc_neighbors(pose))
        sc_neighbor_mask = sc_neighbors >= min_burial
        des_mask *= sc_neighbor_mask

        protein_data["des_mask"] = torch.from_numpy(des_mask).to(torch.bool)
        protein_data["guide_atom_xyz"] = torch.from_numpy(guide_atom_xyz).to(
            torch.float32
        )  # [1, 3]

        # Parse guide sequence
        guide_seq = "X" * n_res if guide_seq is None else guide_seq

        # Seq cond info
        guide_seq = np.array(
            [rc.restype_order.get(aa, rc.restype_num) for aa in guide_seq]
        )
        protein_data["aatype_cond"] = torch.from_numpy(get_seq_cond(guide_seq)).to(
            torch.float32
        )  # [1, 21]
        protein_data["c_idx"] = protein_data["chain_index"]
        return protein_data


class HBDesignerTrainer(SupervisedTrainer):
    def set_default_hps(self, base: TrainConfig) -> None:
        base.model.model_name = "HBDesigner"
        base.model.hbdesigner.max_res = 6
        base.model.hbdesigner.min_res = 2

    def setup(self) -> None:
        super().setup()
        n = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n += p.numel()
        print(f"Number of trainable parameters:\t{n}")

    def setup_data(self) -> None:
        self.train_data = HBDesignerDataset(self.cfg, split="train")
        self.valid_data = HBDesignerDataset(self.cfg, split="valid")

    def build_training_data_loader(self) -> DataLoader:
        return self._make_data_loader(self.train_data)

    def build_validation_data_loader(self) -> DataLoader:
        return self._make_data_loader(self.valid_data)

    def _make_data_loader(
        self,
        src: Dataset,
    ) -> DataLoader:
        return DataLoader(
            dataset=src,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=2 if self.cfg.num_workers else None,
            collate_fn=src.collate,
            worker_init_fn=worker_init,
        )

    def build_test_data_loader(self) -> DataLoader:
        self.test_data = HBDesignerDataset(self.cfg, split="test")
        return self._make_data_loader(self.test_data)

    @torch.no_grad()
    def sample_batch(
        self,
        batch: gd.Batch,
        seq_sample_temp: float = 0.1,
        res_sample_temp: float = 0.1,
    ) -> Tuple[gd.Batch, Dict[int, any]]:
        """
        Sample batch with self.model and update the batch object w/network information.

        Args:
            batch (gd.Batch): Full batch of data for sampling.
            seq_sample_temp (float): Sampling temperature for seq (aatype) decoding.
            res_sample_temp (float): Sampling temperature for res (position/stop action) decoding.

        Returns:
            gd.Batch: Updated batch ready for packing/scoring.
            Dict[int, any]: Dict of raw prediction results.
        """
        self.model.eval()
        results = self.model.sample(
            batch.to(self.cfg.device),
            seq_sample_temp=seq_sample_temp,
            res_sample_temp=res_sample_temp,
        )
        # Apply results to each scaffold in the batch
        b_list = batch.to_data_list()
        for r, b in zip(list(results.keys()), b_list):
            res_idx = results[r]["net_res"]
            res_aatype = results[r]["seq"]
            b["aatype"][:] = rc.restype_order["G"]
            b["aatype"][res_idx] = res_aatype
            b["nll_mask"] = torch.zeros_like(b["aatype"])
            b["nll_mask"][res_idx] = 1.0
        return self.test_data.collate(b_list), results

    def test_batch(
        self,
        b: gd.Batch,
        pack_trainer: SupervisedTrainer,
        seq_sample_temp: float = 0.1,
        res_sample_temp: float = 0.1,
        n_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Sample, pack, and score a Batch of Proteins using HBNet metrics.
        Called by self.test_loop()

        Arguments:
            b (gd.Batch): Batch of proteins to be scored.
            pack_trainer (SupervisedTrainer): HBPackerTrainer object for packing sidechains.
            seq_sample_temp (float): Sampling temperature for restype decoding. Defaults to 0.1.
            res_sample_temp (float): Sampling temperature for network position decoding. Defaults to 0.1.
            n_workers (int): Number of workers for Rosetta parallel packing. Defaults to 1.

        Returns:
            info (Dict[str, any]): Set of metrics for the batch.
        """
        test_info = {}

        # 1. Design sequence
        native_seq = deepcopy(b.aatype_gt)
        native_pos = native_seq != rc.restype_num
        aatype_batch = b.aatype_batch.clone()

        b, results = self.sample_batch(
            b,
            seq_sample_temp=seq_sample_temp,
            res_sample_temp=res_sample_temp,
        )
        pred_seq = b.aatype.clone().to(native_seq.device)
        pred_seq[pred_seq == rc.restype_order["G"]] = rc.restype_num
        pred_pos = pred_seq != rc.restype_num

        # Calculate pos/seq recovery metrics
        pos_rec, seq_rec = calc_seq_rec_batched(
            pred_pos, native_pos, pred_seq, native_seq, aatype_batch
        )
        test_info["pos_rec"] = pos_rec.tolist()
        test_info["seq_rec"] = seq_rec.tolist()

        # Collect avg prob/log-prob of positions
        net_res_probs = []
        seq_probs = []
        for i in range(b.num_graphs):
            net_res_probs.append(np.mean(results[i]["net_res_probs"].tolist()))
            seq_probs.append(np.mean(results[i]["seq_probs"].tolist()))
        test_info["net_res_probs"] = net_res_probs
        test_info["seq_probs"] = seq_probs

        # Need to adjust gd.Batch to be compatible with HBPacker
        pack_ds = pack_trainer.test_data
        b_list = pack_ds.convert_batch_design_to_pack(b, pack_crop=pack_trainer.cfg.model.hbpacker.pack_crop)
        b = pack_ds.collate(b_list)

        # 2. Pack sidechains, if enabled
        proteins, b_list, runtimes = pack_trainer.pack_batch(b, n_workers)
        test_info["pack_time"] = runtimes

        # 3. Loop over each generated protein and score it
        for b_c, p in zip(b_list, proteins):
            if p is None:
                continue

            # Add ghost atom to Protein for scoring + vis
            ghost_atom_xyz = b_c.guide_atom_xyz.numpy()
            p.hetatm_dict = {
                "atom_name": np.array(["V1"]),
                "element": np.array(["V"]),
                "res_name": np.array(["ORI"]),
                "residue_index": np.array([p.residue_index.max() + 1]),
                "chain_index": np.array([p.chain_index.max() + 1]),
                "atom_xyz": ghost_atom_xyz,
            }

            # Score packed networks with HBDesigner metrics
            score_info = score_protein(p)

            # Add network scoring data to test_info
            for key, value in score_info.items():
                if key not in test_info.keys():
                    test_info[key] = [value]
                else:
                    test_info[key].append(value)

        return test_info, proteins

    @torch.no_grad()
    def test_loop(
        self,
        dataloader: DataLoader,
        pack_trainer: SupervisedTrainer,
        seq_sample_temp: float = 0.1,
        res_sample_temp: float = 0.1,
        steps: Optional[int] = None,
        n_workers: int = 1,
        dump: bool = False,
        first_n: int = None,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Run test loop on the specified DataLoader.
        This uses test_batch, which samples, packs, and scores each (predicted) network.
        This is different from SupervisedTrainer.validation_loop(), which only calculates loss.

        Args:
            dataloader (DataLoader): DataLoader of gd.Batch objects to test on.
            pack_trainer (SupervisedTrainer): HBPackerTrainer trainer to use for packing. Must be provided.
            seq_sample_temp (float): Sampling temperature for seq (aatype) decoding.
            res_sample_temp (float): Sampling temperature for res (position/stop action) decoding.
            steps (optional, int): Steps of testing to do. If None, will run whole dataset.
            n_workers (int): Number of workers available for Rosetta packing job. Defaults to 1.
            dump (bool): Whether to dump the packed PDBs after scoring. Defaults to False.
            verbose (bool): Whether to dump per-network stats to CSV. Defaults to False.

        Returns:
            Dict[str, any]: Dictionary of test metrics.
        """
        valid_info = {}
        steps = steps if steps is not None else len(dataloader)
        print(f"Running {steps} test steps...")
        for i, batch in zip(range(steps), cycle(dataloader)):
            batch_info, proteins = self.test_batch(
                batch,
                seq_sample_temp=seq_sample_temp,
                res_sample_temp=res_sample_temp,
                n_workers=n_workers,
                pack_trainer=pack_trainer,
            )
            # Save PDBs to disk, if requested
            if dump:
                for j, p in enumerate(proteins):
                    fname = f"HBDesigner_{i}_{j}.pdb"
                    with open(fname, "w") as fopen:
                        fopen.writelines(p.to_pdb(unk_to_gly=True, no_hetatm=False))

            # Accumulate values from each batch
            if valid_info == {}:
                valid_info = {
                    k: [v] if not isinstance(v, list) else v
                    for k, v in batch_info.items()
                }
            else:
                for k, v in batch_info.items():
                    if isinstance(v, list):
                        valid_info[k].extend(v)
                    else:
                        valid_info[k].append(v)

                n_samples = len(valid_info["pass_strict"])
                print(f"Finished batch {i} of {steps}...({n_samples} / {first_n})")
                print("-" * 50)
                if first_n is not None:
                    if n_samples >= first_n:
                        break

        # Take mean for each metric
        if verbose:
            df = pd.DataFrame.from_dict(valid_info)
            df.to_csv("hbdes_eval_data.csv")

        for k, v in valid_info.items():
            v = np.array(v)
            if first_n is not None:
                v = v[:first_n]
            valid_info[k] = np.nanmean(v)

        valid_info["n_samples"] = first_n
        return valid_info

    def load_model_state(self, ckpt_path: str) -> None:
        # Load weights from saved checkpoint.
        map_location = {"cuda:0": f"cuda:{self.rank}"}
        state = torch.load(ckpt_path, map_location=map_location)

        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)

        # Optimizer and scheduler info.
        self.opt.load_state_dict(state["opt_state_dict"])
        if self.cfg.opt.lr_decay is not None:
            self.lr_sched.load_state_dict(state["lr_sched"])

        # Update initial training step.
        self.initial_step = state["step"]
        if self.initial_step > self.cfg.num_training_steps:
            raise ValueError(
                f"Current initial_step ({self.initial_step}) > num_training_steps ({self.cfg.num_training_steps}!"
            )


def build_hbdesigner_config_longleaf():
    
    config: TrainConfig = init_empty(TrainConfig())
    config.log_dir = f"/work/users/d/i/dieckhau/sandbox/HBDesigner/logs/hbdesigner/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.seed = 42

    # Training settings
    config.validate_every = 1_024
    config.num_validation_batches = 128
    config.checkpoint_every = 50_000
    config.num_training_steps = 200_000
    config.num_workers = 16
    config.print_every = 16

    # Loss and learning rate settings
    config.opt.opt = "noam"
    config.opt.adam_eps = 1e-9
    config.opt.lr_decay = None
    config.opt.noam_factor = 0.5

    config.model.hbdesigner.batch_size = 10_000
    config.model.hbdesigner.loss_type = "focal"
    config.model.hbdesigner.focal_gamma = 2.0
    config.model.hbdesigner.seq_nll_weight = 1.0
    config.model.hbdesigner.net_res_nll_weight = 1.0

    # Conditioning info
    config.model.hbdesigner.guide_atom_pct = 0.5
    config.model.hbdesigner.guide_atom_sigma = 4.0
    config.model.hbdesigner.seq_cond_pct = 0.2
    config.model.hbdesigner.seq_cond_unk_pct = 0.5

    # Other model settings
    config.model.hbdesigner.bb_noise = 0.2
    config.model.model_name = "HBDesigner"
    config.model.hbdesigner.data_location = (
        "/work/users/d/i/dieckhau/pdb_2021aug02_hbdesigner4/"
    )

    return config


CONFIGS = {
    "hbdesigner_longleaf": build_hbdesigner_config_longleaf,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hbdesigner_longleaf")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_id", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    # Validate resume arguments
    if args.resume_id is None != args.resume_ckpt is None:
        raise ValueError(
            "If resume_id is provided, resume_ckpt must also be provided. And vice versa."
        )

    # Set up the trainer.
    config = CONFIGS[args.config]()
    seed_everything(config.seed)
    trainer = HBDesignerTrainer(config)

    # Load checkpoint if resuming
    if args.resume_ckpt:
        assert os.path.isfile(args.resume_ckpt), (
            f"Invalid checkpoint file {args.resume_ckpt} specified."
        )
    # Set up wandb.
    if args.use_wandb:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        if not args.resume_id:
            wandb.init(project="HBDesigner", config=OmegaConf.to_container(trainer.cfg))
        else:
            wandb.init(
                project="HBDesigner",
                config=OmegaConf.to_container(trainer.cfg),
                id=args.resume_id,
                resume="must",
            )

    # Fit HBDesigner
    trainer.fit()
