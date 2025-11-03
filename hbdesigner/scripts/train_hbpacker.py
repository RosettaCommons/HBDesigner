import argparse
import os
from copy import deepcopy
import datetime
from typing import Any, Dict, Optional, Sequence, Tuple
import time
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch_geometric.data as gd
import wandb
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist


import hbdesigner.data.residue_constants as rc
from hbdesigner.data.features import (
    calc_bb_dihedrals,
    impute_CB,
    calc_sc_dihedrals,
    build_sc_from_chi,
    sincos_to_angle,
)
from hbdesigner.data.hbnet import (
    batch_to_proteins,
    pack_with_rosetta,
    rosetta_hbond_detect,
    calc_seq_rec_batched,
    clear_non_network_res,
    crop_by_distance,
)
from hbdesigner.model.pippack_model import (
    apply_logits_to_proteins,
    get_sidechain_logits,
    load_PIPPack,
)
from hbdesigner.data.protein import Protein
from hbdesigner.scripts.preprocess_asmbs import load_metadata
from hbdesigner.train.config import TrainConfig
from hbdesigner.train.trainer import SupervisedTrainer
from hbdesigner.utils import cycle, seed_everything, worker_init, init_empty
from hbdesigner.scripts.train_hbdesigner import HBDesignerDataset


class HBPackerDataset(HBDesignerDataset):
    """
    Dataset for HPacker model.

    Iterates over clusters and samples a random assembly, then collects any valid networks.
    Randomly samples from candidate networks according to config params.

    Packer dataset featurizes the fully decoded network w/random initial sidechains.

    Expected dataset format:
        - ABCD.pt files for all PDBs
        - ABCD_N.npz files for all assemblies
        - ABCD_N.gml files for all assemblies
        - ABCD_N_hbnet.npz files for all HBNet designs
    """

    def __init__(self, cfg: TrainConfig, split: str = "train"):
        super().__init__(cfg, split)

    def featurize(self, p: Protein, hbnet_arr: np.ndarray) -> gd.Data:
        """
        Featurize Protein and Graph info into HBPacker model inputs.

        Arguments:
            p (Protein): Protein object.
            hbnet_arr (np.ndarray): Copy of p.aatype with all non network residues set to GLY.

        Returns:
            gd.Data: torch_geometric Data object with featurized protein.
        """

        c = self.cfg.model.hbpacker
        # Override seq for HBNet
        hbnet_pos = np.where(hbnet_arr != rc.restype_order["G"])[0]

        if (hbnet_pos.size > c.max_res) or (hbnet_pos.size < c.min_res):
            return

        # Check for invalid restypes in network res
        invalid = hbnet_arr[hbnet_pos][:, None] == rc.restype_non_hb_idx[None, :]
        if np.sum(invalid) > 0.0:
            return

        # Crop if requested
        if c.pack_crop > 0.0:
            p, knn = crop_by_distance(p, hbnet_pos, c.pack_crop)
            hbnet_arr = hbnet_arr[knn]
            hbnet_pos = np.where(hbnet_arr != rc.restype_order["G"])[0]

        # Chi mask includes all decoded network res
        chi_mask = np.zeros_like(p.aatype, dtype=np.int32)
        chi_mask[hbnet_pos] = 1
        chi_mask_bool = chi_mask > 0

        # Clear any non-network res for ground-truth aatype and xyz
        p.aatype[~chi_mask_bool] = rc.restype_num
        p.aatype[hbnet_pos] = hbnet_arr[hbnet_pos]
        aatype = p.aatype
        p.atom27_xyz[~chi_mask_bool, 4:] = 0.0
        p.atom27_mask[~chi_mask_bool, 4:] = 0
        atom14_xyz_gt = p.atom27_xyz[:, :14]
        atom14_mask_gt = p.atom27_mask[:, :14]
        residue_index = p.residue_index
        chain_index = p.chain_index

        # Calculate ground truth chi dihedrals and their sin and cos.
        sc_dihedral_gt, sc_dihedral_mask_gt = calc_sc_dihedrals(
            atom14_xyz_gt[:, :14], aatype, return_mask=True
        )

        chi_sincos_gt = np.stack(
            [np.sin(sc_dihedral_gt), np.cos(sc_dihedral_gt)], axis=-1
        )

        # End of ground-truth features
        p = deepcopy(p)
        # Clear ALL sidechains from input feats
        p.atom27_xyz[:, 4:] = 0.0
        p.atom27_mask[:, 4:] = 0.0
        atom14_xyz = p.atom27_xyz[:, :14]
        atom14_mask = p.atom27_mask[:, :14]

        # Zero out starting dihedrals and get xyz to match
        sc_dihedral = np.zeros_like(sc_dihedral_gt)
        sc_dihedral_mask = np.array(rc.chi_angles_mask)[aatype]  # [L, 4]
        atom14_xyz_sc, atom14_mask_sc = build_sc_from_chi(
            atom14_xyz_gt[:, :4], aatype, sc_dihedral, sc_dihedral_mask
        )
        atom14_xyz[chi_mask_bool] = atom14_xyz_sc[chi_mask_bool]
        atom14_mask[chi_mask_bool] = atom14_mask_sc[chi_mask_bool]

        # Noise bb xyz to avoid seq rec cheating
        if (self.split == "train") and (c.bb_noise > 0.0):
            atom14_xyz[..., :5, :] = atom14_xyz[..., :5, :] + (
                self.cfg.model.hbpacker.bb_noise
                * np.random.randn(*atom14_xyz[..., :5, :].shape)
            )

        # Use noised xyz to get bb dihedrals
        bb_dihedral = calc_bb_dihedrals(atom14_xyz, p.residue_index, return_mask=False)

        # Create the Data object
        protein_data = gd.Data(
            num_nodes=aatype.shape[0],
            x=torch.zeros((1, 1)),
            # Ground truth sidechains
            atom14_xyz_gt=torch.from_numpy(atom14_xyz_gt).to(
                torch.float32
            ),  # [L, 14, 3]
            atom14_mask_gt=torch.from_numpy(atom14_mask_gt).to(
                torch.float32
            ),  # [L, 14]
            # Input seq and sidechains
            aatype=torch.from_numpy(aatype).to(torch.long),  # [L]
            atom14_xyz=torch.from_numpy(atom14_xyz).to(torch.float32),  # [L, 14, 3]
            atom14_mask=torch.from_numpy(atom14_mask).to(torch.float32),  # [L, 14]
            residue_index=torch.from_numpy(residue_index).to(torch.int32),  # [L]
            chain_index=torch.from_numpy(chain_index).to(torch.int32),  # [L]
            bb_dihedral=torch.from_numpy(bb_dihedral).to(torch.float32),  # [L, 3]
            sc_dihedral=torch.from_numpy(sc_dihedral).to(torch.float32),  # [L, 4]
            # Ground truth sidechain data
            sc_dihedral_mask_gt=torch.from_numpy(sc_dihedral_mask_gt).to(
                torch.float32
            ),  # [L, 4]
            chi_sincos_gt=torch.from_numpy(chi_sincos_gt).to(
                torch.float32
            ),  # [L, 4, 2]
            # Masks and cond info
            chi_mask=torch.from_numpy(chi_mask).to(torch.float32),  # [L]
        )

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
                (batch.chi_mask[batch.aatype_batch == j]).sum()
                for j in range(batch.num_graphs)
            ],
        ).long()  # [B,]
        return batch

    # @staticmethod
    # def featurize_inference(
    #     p: Protein,
    #     n_res: int = 2,
    #     guide_res: np.ndarray = None,
    #     guide_radius: float = 1e6,
    #     guide_seq: str = None,
    #     min_burial: float = 0.0,
    # ) -> gd.Data:
    #     """
    #     Featurize Protein for inference without available reference HBNet.

    #     Arguments:
    #         p (Protein): Protein object.
    #         n_res (int): Number of residues to include in predicted network. Default is 2.
    #         guide_res (np.ndarray): Array of positions for inclusion in guide atom centroid calculation. Default is None (ignored).
    #         guide_radius (float): Radius around the guide atom to allow designable. Default is 1e6 (all residues).
    #         guide_seq (str): Guide sequence to enforce in all designs. Default is None (all UNK).
    #         min_burial (float): Minimum burial value to allow designable. Default is 0.0. Core is 5.2, Surface is 2.0.

    #     Returns:
    #         gd.Data: torch_geometric Data object with featurized protein.
    #     """
    #     # Protein features
    #     p.clear_sequence()
    #     aatype = p.aatype
    #     atom14_xyz = p.atom27_xyz[:, :14]
    #     atom14_mask = p.atom27_mask[:, :14]
    #     residue_index = p.residue_index
    #     chain_index = p.chain_index
    #     bb_dihedral = calc_bb_dihedrals(
    #         p.atom27_xyz[:, :14], p.residue_index, return_mask=False
    #     )

    #     # Zero out starting dihedrals and get xyz to match
    #     sc_dihedral = np.zeros((p.n_res, 4), dtype=np.float32)  # [L, 4]
    #     sc_dihedral_mask = np.array(rc.chi_angles_mask)[aatype]  # [L, 4]

    #     # Create the Data object
    #     protein_data = gd.Data(
    #         num_nodes=aatype.shape[0],
    #         x=torch.zeros((1, 1)),  # x is used often to identify the device
    #         aatype=torch.from_numpy(aatype).to(torch.long),  # [L]
    #         atom14_xyz=torch.from_numpy(atom14_xyz).to(torch.float32),  # [L, 14, 3]
    #         atom14_mask=torch.from_numpy(atom14_mask).to(torch.float32),  # [L, 14]
    #         residue_index=torch.from_numpy(residue_index).to(torch.int32),  # [L]
    #         chain_index=torch.from_numpy(chain_index).to(torch.int32),  # [L]
    #         bb_dihedral=torch.from_numpy(bb_dihedral).to(torch.float32),  # [L, 3]
    #         # Empty dihedrals prior to design
    #         sc_dihedral=torch.from_numpy(sc_dihedral).to(torch.float32),  # [L, 4]
    #         sc_dihedral_mask=torch.from_numpy(sc_dihedral_mask).to(
    #             torch.float32
    #         ),  # [L, 4]
    #     )

    #     # nll_mask is mask of designable positions
    #     nll_mask = np.ones_like(p.aatype, dtype=np.int32)
    #     protein_data["nll_mask"] = torch.from_numpy(nll_mask).to(torch.float32)
    #     protein_data["aatype_masked"] = torch.from_numpy(p.aatype).to(torch.long)

    #     # done_mask is mask of already-designed positions
    #     done_mask = np.zeros_like(p.aatype, np.int32)
    #     protein_data["done_mask"] = torch.from_numpy(done_mask).to(torch.long)

    #     # Guide atom cond info
    #     guide_atom_sigma = 4.0  # sd of guide atom sampling distribution
    #     if guide_res is None:
    #         guide_atom_xyz = np.zeros_like(
    #             atom14_xyz[0, 0:1, :],
    #         )
    #         des_mask = np.prod(atom14_mask[..., :4], axis=-1)
    #     else:
    #         guide_atom_xyz = get_guide_atom(
    #             atom14_xyz[guide_res, :3, :], guide_atom_sigma
    #         )
    #         # Make mask of nearby residues
    #         cb_xyz = impute_CB(
    #             atom14_xyz[..., 0, :], atom14_xyz[..., 1, :], atom14_xyz[..., 2, :]
    #         )
    #         cb_dist = np.squeeze(cdist(cb_xyz, guide_atom_xyz))  # [L]
    #         des_mask = (cb_dist <= guide_radius) * np.prod(
    #             atom14_mask[..., :3], axis=-1
    #         )

    #     # Add burial constraint to designable positions
    #     pose = Pose()
    #     pose_from_pdbstring(pose, p.to_pdb(unk_to_gly=True))
    #     sc_neighbors = np.array(calc_sc_neighbors(pose))
    #     sc_neighbor_mask = sc_neighbors >= min_burial
    #     des_mask *= sc_neighbor_mask

    #     protein_data["des_mask"] = torch.from_numpy(des_mask).to(torch.bool)
    #     protein_data["guide_atom_xyz"] = torch.from_numpy(guide_atom_xyz).to(
    #         torch.float32
    #     )  # [1, 3]

    #     # Parse guide sequence
    #     guide_seq = "X" * n_res if guide_seq is None else guide_seq

    #     # Seq cond info
    #     guide_seq = np.array(
    #         [rc.restype_order.get(aa, rc.restype_num) for aa in guide_seq]
    #     )
    #     protein_data["aatype_cond"] = torch.from_numpy(get_seq_cond(guide_seq)).to(
    #         torch.float32
    #     )  # [1, 21]
    #     protein_data["c_idx"] = protein_data["chain_index"]
    #     return protein_data

    # @staticmethod
    # def featurize_inference_packing(
    #     p: Protein,
    #     hbnet_pos: np.ndarray,
    #     hbnet_res: np.ndarray,
    #     pack_crop: float = 10.0,
    # ) -> gd.Data:
    #     """
    #     Featurize Protein for inference without available reference HBNet.

    #     Arguments:
    #         p (Protein): Protein object.
    #         hbnet_pos (np.ndarray): Array of positions for inclusion in predicted network.
    #         hbnet_res (np.ndarray): Array of residues for inclusion in predicted network.
    #         pack_crop (float): Distance in Angstroms to crop the protein around the network. Default is 10.0.

    #     Returns:
    #         gd.Data: torch_geometric Data object with featurized protein.
    #     """
    #     # Protein features
    #     p.clear_sequence()
    #     p.aatype[hbnet_pos] = hbnet_res

    #     # Crop before featurizing
    #     if pack_crop > 0.:
    #         p, knn = crop_by_distance(p, hbnet_pos, pack_crop)
    #         hbnet_pos = np.where(p.aatype != rc.restype_num)[0]
    #     else:
    #         knn = np.arange(p.n_res)

    #     aatype = p.aatype
    #     atom14_xyz = p.atom27_xyz[:, :14]
    #     atom14_mask = p.atom27_mask[:, :14]
    #     residue_index = p.residue_index
    #     chain_index = p.chain_index
    #     bb_dihedral = calc_bb_dihedrals(
    #         p.atom27_xyz[:, :14], p.residue_index, return_mask=False
    #     )

    #     # Zero out starting dihedrals and get xyz to match
    #     sc_dihedral = np.zeros((p.n_res, 4), dtype=np.float32)  # [L, 4]
    #     sc_dihedral_mask = np.array(rc.chi_angles_mask)[aatype]  # [L, 4]

    #     # Zero out starting dihedrals and get xyz to match
    #     atom14_xyz_sc, atom14_mask_sc = build_sc_from_chi(
    #         atom14_xyz[:, :4], aatype, sc_dihedral, sc_dihedral_mask
    #     )
    #     atom14_xyz[hbnet_pos] = atom14_xyz_sc[hbnet_pos]
    #     atom14_mask[hbnet_pos] = atom14_mask_sc[hbnet_pos]

    #     # Create the Data object
    #     protein_data = gd.Data(
    #         num_nodes=aatype.shape[0],
    #         x=torch.zeros((1, 1)),  # x is used often to identify the device
    #         aatype=torch.from_numpy(aatype).to(torch.long),  # [L]
    #         atom14_xyz=torch.from_numpy(atom14_xyz).to(torch.float32),  # [L, 14, 3]
    #         atom14_mask=torch.from_numpy(atom14_mask).to(torch.float32),  # [L, 14]
    #         residue_index=torch.from_numpy(residue_index).to(torch.int32),  # [L]
    #         chain_index=torch.from_numpy(chain_index).to(torch.int32),  # [L]
    #         bb_dihedral=torch.from_numpy(bb_dihedral).to(torch.float32),  # [L, 3]
    #         # Empty dihedrals prior to design
    #         sc_dihedral=torch.from_numpy(sc_dihedral).to(torch.float32),  # [L, 4]
    #         sc_dihedral_mask=torch.from_numpy(sc_dihedral_mask).to(
    #             torch.float32
    #         ),  # [L, 4]
    #         pack_knn=torch.from_numpy(knn).to(torch.long),  # [K]
    #     )

    #     # nll_mask is mask of designable positions
    #     nll_mask = np.ones_like(p.aatype, dtype=np.int32)
    #     protein_data["nll_mask"] = torch.from_numpy(nll_mask).to(torch.float32)
    #     protein_data["aatype_masked"] = torch.from_numpy(p.aatype).to(torch.long)

    #     # done_mask is mask of already-designed positions
    #     done_mask = np.zeros_like(p.aatype, np.int32)
    #     protein_data["done_mask"] = torch.from_numpy(done_mask).to(torch.long)

    #     # chi nll mask is mask of packable positions
    #     protein_data["chi_nll_mask"] = torch.from_numpy(p.aatype != rc.restype_num).to(torch.float32)

    #     protein_data["c_idx"] = protein_data["chain_index"]
    #     return protein_data


class HBPackerTrainer(SupervisedTrainer):
    def set_default_hps(self, base: TrainConfig) -> None:
        base.model.model_name = "HBPacker"
        base.model.hbpacker.max_res = 6
        base.model.hbpacker.min_res = 2

    def setup(self) -> None:
        super().setup()
        n = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n += p.numel()
        print(f"Number of trainable parameters:\t{n}")

    def setup_data(self) -> None:
        c = self.cfg.model.hbpacker
        if c.pack_method == "pippack":
            print(f"Loading PIPPack ({c.pack_mode})!")
            if c.pack_mode == "fast":
                self.cfg.model.frankenpacker.pippack_recycles = 1
                pippack_models = 3
            else:
                self.cfg.model.frankenpacker.pippack_recycles = 3
                pippack_models = 3
            self.load_pippack(n_models=pippack_models)

        self.train_data = HBPackerDataset(self.cfg, split="train")
        self.valid_data = HBPackerDataset(self.cfg, split="valid")

    def load_pippack(self, n_models: int = 1) -> None:
        """
        Loads one or more PIPPack models for packing use. Stored as a list in the self.pippack attribute.

        Args:
            n_models (int): How many models to load for ensembling. Default is 1.

        """
        self.pippack = []
        models = ["1", "2", "3"][:n_models]
        for m in models:
            ckpt = self.cfg.model.frankenpacker.pippack_ckpt
            ckpt = "_".join(ckpt.split("_")[:-2]) + f"_{m}_ckpt.pt"
            self.cfg.model.frankenpacker.pippack_ckpt = ckpt
            mod = load_PIPPack(self.cfg.model)
            mod.eval()
            self.pippack.append(mod)

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
        self.test_data = HBPackerDataset(self.cfg, split="test")
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
            b["chi_nll_mask"][res_idx] = 1.0
        return self.test_data.collate(b_list), results

    def test_batch(
        self,
        b: gd.Batch,
        design: bool = True,
        pack: bool = True,
        seq_sample_temp: float = 0.1,
        res_sample_temp: float = 0.1,
        n_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Sample, pack, and score a Batch of Proteins using HBNet metrics.
        Called by self.test_loop()

        Arguments:
            b (gd.Batch): Batch of proteins to be scored.
            design (bool): Whether to design with HBDes3. Default is True. If False, native seq is used.
            pack (bool): Whether to pack sidechains. Default is True. If False, ???.
            seq_sample_temp (float): Sampling temperature for restype decoding. Defaults to 0.1.
            res_sample_temp (float): Sampling temperature for network position decoding. Defaults to 0.1.
            n_workers (int): Number of workers for Rosetta parallel packing. Defaults to 1.

        Returns:
            info (Dict[str, any]): Set of metrics for the batch.
        """
        test_info = {}

        # 1. Design seq, or collect native seq
        if design:
            native_seq = deepcopy(b.aatype_gt)
            native_pos = native_seq != rc.restype_num
            aatype_batch = b.aatype_batch.clone()
            # done_before = b.done_mask.clone() # TODO drop, partial network recovery
            b, results = self.sample_batch(
                b,
                seq_sample_temp=seq_sample_temp,
                res_sample_temp=res_sample_temp,
            )
            pred_seq = b.aatype.clone().to(native_seq.device)
            pred_seq[pred_seq == rc.restype_order["G"]] = rc.restype_num
            pred_pos = pred_seq != rc.restype_num

            # TODO drop, partial network recovery
            # mask = (done_before > 0).cpu()
            # native_pos = native_pos[~mask]
            # native_seq = native_seq[~mask]
            # pred_pos = pred_pos[~mask]
            # pred_seq = pred_seq[~mask]
            # aatype_batch = aatype_batch[~mask]

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

        else:
            # Override preds with native seq
            b.aatype = b.aatype_gt
            b.nll_mask = b.nll_mask + b.done_mask

        # 2. Pack sidechains, if enabled
        proteins, b_list, runtimes = self.pack_batch(b, n_workers)
        test_info["pack_time"] = runtimes

        # 3. Loop over each generated protein and score it
        for b_c, p in zip(b_list, proteins):
            if p is None:
                continue

            # Calculate packing scores, only if running packing eval
            if (not design) and pack:
                pack_info = self.compute_packing_metrics(p, b_c)
                # Collect packing metrics
                for key, value in pack_info.items():
                    x = value.cpu().tolist()
                    if key not in test_info.keys():
                        test_info[key] = x
                    else:
                        test_info[key].extend(x)

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

            # In all cases, score packed networks
            score_info = self.score_protein(p)

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
        design: bool = True,
        pack: bool = True,
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
        This uses test_batch, which packs and scores each (predicted) network.
        This is different from SupervisedTrainer.validation_loop(), which only calculates loss.

        Args:
            dataloader (DataLoader): DataLoader of gd.Batch objects to test on.
            design (bool): Whether to design a new network instead of using the native. Defaults to True.
            pack (bool): Whether to pack the sequence instead of using native sidechains. Defaults to True.
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
                design=design,
                pack=pack,
                seq_sample_temp=seq_sample_temp,
                res_sample_temp=res_sample_temp,
                n_workers=n_workers,
            )
            # Save PDBs to disk, if requested
            if dump:
                for j, p in enumerate(proteins):
                    fname = f"HBDes3_{i}_{j}.pdb"
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

    def pack_batch(self, b: gd.Batch, n_workers: int = 1) -> Sequence[Protein]:
        """
        Pack a Batch of data by sending it to the packing method specified in the config.
        Note that this uses the batch masks to detect network residues.

        Args:
            batch: (gd.Batch): Batch of data from HBDesignerDataset.collate.
            n_workers (int): Number of workers for Rosetta parallel packing. Defaults to 1.

        Returns:
            Sequence[Protein]: List of packed proteins.
        """
        c = self.cfg.model.hbdesigner
        b = b.to("cpu")
        if c.pack_method == "native":
            # Override w/native coords
            b.atom14_xyz = b.atom14_xyz_gt
            b.atom14_mask = b.atom14_mask_gt
            b.aatype = b.aatype_gt
            proteins, b_list = batch_to_proteins(b)

            if c.pack_min:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    mode="minimize-cart",
                )
            else:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    # mode="pdb2pqr",
                    mode="reduce",
                )
                print("ran reduce/hydride/pdb2pqr!")

        elif c.pack_method == "rosetta":
            proteins, b_list = batch_to_proteins(b)

            # Prep for packing by clearing non-network residues
            for p, b in zip(proteins, b_list):
                mask = (b.aatype != rc.restype_order["G"]) * (
                    b.aatype != rc.restype_num
                )
                p = clear_non_network_res(p, b, mask, unk="G")

            # Separated pack + minimization operations
            proteins = pack_with_rosetta(
                proteins,
                n_workers=n_workers,
                mode="pack",
            )
            if c.pack_min:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    mode="minimize",
                )

        elif c.pack_method == "hbdes3":
            self.pack_model.eval()
            t0 = time.time()
            b.sc_dihedral_mask = b.sc_dihedral_mask_gt
            b.aatype[b.aatype == rc.restype_order["G"]] = rc.restype_num
            b.chi_nll_mask = (b.aatype != rc.restype_num).to(torch.long)
            b, results = self.pack_model.run_pack_recyc(
                b.to(self.cfg.device), c.num_recycles
            )
            non_net = b.aatype == 20
            b.atom14_xyz[non_net, 4:] = 0.0
            b.atom14_mask[non_net, 4:] = 0.0
            # Convert to proteins and plist after pack
            proteins, b_list = batch_to_proteins(b)
            t1 = time.time()
            rtime = (t1 - t0) / b.num_graphs
            for p in proteins:
                p.pack_time = rtime

            if c.pack_min:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    mode="minimize-cart",
                )
            else:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    mode="reduce",
                )
                print("ran reduce/hydride!")

        elif c.pack_method == "pippack":
            # Using Frankenpacker PIPPack configuration
            t0 = time.time()
            proteins, b_list = batch_to_proteins(b)
            proteins = self.pack_with_pippack(proteins, b_list)
            t1 = time.time()
            rtime = (t1 - t0) / b.num_graphs
            for p in proteins:
                p.pack_time = rtime

            if c.pack_min:
                proteins = pack_with_rosetta(
                    proteins,
                    n_workers=n_workers,
                    mode="minimize-cart",
                )
        else:
            raise ValueError(f"Invalid pack_method {c.pack_method} provided")
        # Calculate per-protein runtime
        try:
            runtimes = [p.pack_time for p in proteins]
        except AttributeError:
            runtimes = [0.0 for p in proteins]

        return proteins, b_list, runtimes

    def score_protein(self, p: Protein) -> Dict[str, float]:
        """
        Score an individual Protein for HBond metrics.

        Args:
            p (Protein): Stripped (repacked) Protein ready for network analysis.

        Returns:
            Dict[str, float]: Dict of single protein metrics and their labels.
        """

        # Get bonds and energy metrics from Rosetta
        bond_list, rosetta_stats = rosetta_hbond_detect(
            p,
            max_energy=0.0,
            optH=True,  # True
            optH_MCA=False,
        )
        print("bond list rosetta:", bond_list, "***")

        # bond_list, rosetta_stats = biotite_hbond_detect(
        #     p,
        # )
        # print("bond list biotite:", bond_list, '***')

        # Save guide atom before cropping
        guide_atom_xyz = np.copy(p.hetatm_dict["atom_xyz"])

        # Drop ghost residues, which can affect chain counting
        res_mask = np.prod(p.atom27_mask[:, :4], axis=-1)
        p = p.mask(np.where(res_mask)[0])

        # Drop non-network res
        hb_idx = np.where(
            (p.aatype != rc.restype_order["G"]) * (p.aatype != rc.restype_num)
        )[0]

        # Check if all chains are used, as they should be
        chains_total = np.unique(p.chain_index).size
        p = p.mask(hb_idx)
        chains_used = np.unique(p.chain_index).size
        used_all_chains = float(chains_used == chains_total)

        # Count motif nodes and edges
        hbond_res = []
        for b in bond_list:
            hbond_res.extend(list(b))
        hbond_res = np.unique(hbond_res).size
        hbond_bonds = len(bond_list)
        total_res = p.n_res
        hbond_res_frac = hbond_res / total_res

        # Calculate max pairwise distance b/w motif Cb atoms
        bb_xyz = p.atom27_xyz[:, :4, :]
        cb_xyz = impute_CB(bb_xyz[:, 0], bb_xyz[:, 1], bb_xyz[:, 2])
        cd = cdist(cb_xyz, cb_xyz)
        max_Cb_dist = np.max(np.triu(cd))

        # Check if residue aatypes are valid
        valid_res = np.array([rc.restype_order[r] for r in rc.restype_hb_sc])
        vmask = np.sum(p.aatype == valid_res[:, None], axis=0) > 0
        valid_res_frac = np.sum(vmask) / total_res

        # Calculate network displacement vs guide atom
        xyz_Cb = impute_CB(
            p.atom27_xyz[:, 0, :], p.atom27_xyz[:, 1, :], p.atom27_xyz[:, 2, :]
        )  # [N, 3]
        xyz_centroid = np.mean(xyz_Cb, axis=0)  # [3]
        displacement = np.linalg.norm(guide_atom_xyz - xyz_centroid)

        # Check if all residues are h-bonding
        pass_relaxed = hbond_res == total_res
        # Check if network has only one connected component
        g = nx.Graph()
        g.add_edges_from(bond_list)
        n_comp = len(list(nx.connected_components(g)))
        pass_strict = pass_relaxed and (n_comp == 1)

        score_info = {
            "pass_strict": float(pass_strict),
            "pass_relaxed": float(pass_relaxed),
            "hbond_res_frac": hbond_res_frac,
            "hbond_res": hbond_res,
            "hbond_bonds": hbond_bonds,
            "chains_used": float(chains_used),
            "used_all_chains": used_all_chains,
            "total_res": total_res,
            "max_Cb_dist": max_Cb_dist,
            "valid_res_frac": valid_res_frac,
            "displacement": displacement,
        }
        score_info.update(rosetta_stats)

        # Collect restype fractions
        for aa in rc.restypes:
            score_info[aa + "_frac"] = (
                np.sum(p.aatype == rc.restype_order[aa]) / total_res
            )
        return score_info

    def compute_packing_metrics(self, p: Protein, b: gd.Data) -> Dict[str, Any]:
        pack_info = {}
        # Collect true and pred chi values
        true_chi_rad = sincos_to_angle(b.chi_sincos_gt)
        p_mask = p.atom27_mask[:, 5] != 0
        pred_coords = torch.from_numpy(p.atom27_xyz[p_mask, :14]).to(torch.float32)
        pred_chi_rad = calc_sc_dihedrals(
            pred_coords, torch.from_numpy(p.aatype[p_mask]), return_mask=False
        )

        # Collect relevant chi mask
        mask = (b.nll_mask + b.done_mask).bool()
        chi_mask = b.sc_dihedral_mask_gt[mask]
        dev = next(self.model.parameters()).device

        # Compute metrics on GPU so we can use model convenience methods
        chi_ae = self.model.compute_chi_ae(
            pred_chi_rad=pred_chi_rad.to(dev).to(torch.float32),
            aatype=b.aatype_gt[mask].to(dev).to(torch.long),
            chi_mask=chi_mask.to(dev).to(torch.float32),
            true_chi_rad=true_chi_rad[mask].to(dev).to(torch.float32),
        )
        for i_chi, chi_error in enumerate(torch.unbind(chi_ae, -1)):
            pack_info[f"chi_mae_chi_{i_chi + 1}"] = chi_error[
                chi_mask[:, i_chi].bool()
            ].cpu() * (180.0 / np.pi)

        rad_thresh = (20 / 180) * torch.pi
        correct_chi = chi_mask * (chi_ae.cpu() - rad_thresh) < 0.0
        rot_rec = (correct_chi.sum(-1) == chi_mask.sum(-1)).float()
        pack_info["rot_rec"] = rot_rec.cpu()

        # Update batch xyz with protein xyz
        b.atom14_xyz[mask] = pred_coords.to(b.x.device)

        pack_info["sc_rmsd"] = (
            self.pack_model.compute_sc_msd(
                b.atom14_xyz[mask],
                b.atom14_xyz_gt[mask],
                b.atom14_mask_gt[mask],
                b.aatype_gt[mask],
            )
            .sqrt()
            .cpu()
        )

        # Collapse pack info into per-protein statistics
        for k, v in pack_info.items():
            pack_info[k] = torch.nanmean(v, keepdim=True)
        return pack_info

    @torch.no_grad()
    def pack_with_pippack(
        self,
        proteins: Sequence[Protein],
        b_list: Sequence[gd.Data],
    ) -> Sequence[Protein]:
        """
        Pack a full Protein object.

        Args:
            proteins (Sequence[Protein]): List of Protein objects for packing.
            b_list (Sequence[gd.Data]): List of gd.Data objects (not used).

        Returns:
            Sequence[Protein]: List of repacked Proteins.

        """
        assert self.pippack is not None, (
            "Error: you need to load PIPPack before using it!"
        )
        # Prep batch for PIPPack
        p_batch = []
        for p in proteins:
            p.aatype[p.aatype == rc.restype_num] = rc.restype_order["G"]
            p.clear_sidechains()
            p_b = self.test_data.featurize(p, p.aatype)
            p_b["aatype"] = p_b["aatype_gt"]
            p_b["nll_mask"] = p_b["nll_mask"] + p_b["done_mask"]
            p_b["chi_nll_mask"] = p_b["nll_mask"]

            # PolyALA backbone works best
            p_b["aatype"][p_b["aatype"] == rc.restype_num] = rc.restype_order["A"]
            p_b["atom14_xyz"][:, 4] = impute_CB(
                p_b["atom14_xyz"][:, 0],
                p_b["atom14_xyz"][:, 1],
                p_b["atom14_xyz"][:, 2],
            )
            p_b["atom14_mask"][:, 4] = torch.prod(p_b["atom14_mask"][:, :4], dim=-1)

            # Zero out sc dihedrals and get new dihedral masks from aatype
            netres = np.where(p_b["aatype"] != rc.restype_order["A"])[0]
            p_b["atom14_xyz"][netres, 5:] += (
                1e-4  # make nonzero to get correct dihedral masks
            )
            p_b["sc_dihedral"], p_b["sc_dihedral_mask"] = calc_sc_dihedrals(
                p_b["atom14_xyz"][:, :14], p_b["aatype"], return_mask=True
            )
            p_b["sc_dihedral"] = torch.zeros_like(p_b["sc_dihedral"])

            # Rebuild xyz based on zeroed dihedrals
            atom14_xyz_sc, atom14_mask_sc = build_sc_from_chi(
                p_b["atom14_xyz"][netres, :4],
                p_b["aatype"][netres],
                p_b["sc_dihedral"][netres],
                p_b["sc_dihedral_mask"][netres],
            )
            p_b["atom14_xyz"][netres] = atom14_xyz_sc
            p_b["atom14_mask"][netres] = atom14_mask_sc
            p_batch.append(p_b)

        p_batch = self.test_data.collate(p_batch)

        # Get and apply PIPPack preds
        all_logits = []
        for m in self.pippack:
            logits = (
                get_sidechain_logits(
                    m,
                    p_batch.to(self.cfg.model.frankenpacker.pippack_device),
                    recycles=self.cfg.model.frankenpacker.pippack_recycles,
                )
                .detach()
                .cpu()
            )
            all_logits.append(logits)

        # Stack and avg logits from ensemble
        logits = torch.mean(torch.stack(all_logits, dim=-1), dim=-1)
        proteins = apply_logits_to_proteins(
            proteins, logits, resample=self.cfg.model.frankenpacker.pippack_resampling
        )
        for p in proteins:
            p.aatype[p.aatype == rc.restype_order["A"]] = rc.restype_order["G"]
        return proteins

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


def build_hbpacker_config_longleaf():
    config: TrainConfig = init_empty(TrainConfig())
    config.log_dir = f"/work/users/d/i/dieckhau/sandbox/HBDesigner/logs/hbpacker/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.seed = 42

    # Training settings
    config.validate_every = 1_024
    config.num_validation_batches = 128
    config.checkpoint_every = 50_000
    config.num_training_steps = 50_000
    config.num_workers = 16
    config.print_every = 16

    # Loss and learning rate settings (optimized)
    config.opt.opt = "noam"
    config.opt.adam_eps = 1e-9
    config.opt.lr_decay = None
    config.opt.noam_factor = 0.5

    # Other model settings
    config.model.hbpacker.bb_noise = 0.02
    config.model.model_name = "HBPacker"
    config.model.hbpacker.data_location = (
        "/work/users/d/i/dieckhau/pdb_2021aug02_hbdesigner4/"
    )
    config.model.hbpacker.batch_size = 10_000

    # Packing config
    config.model.hbpacker.sc_msd_weight = 1.0
    config.model.hbpacker.chi_mse_weight = 1.0
    config.model.hbpacker.chi_norm_weight = 0.1

    config.model.hbpacker.sc_clash_weight = 0.2
    config.model.hbpacker.orient_msd_weight = 0.2
    config.model.hbpacker.reweight_chi_mse = True
    config.model.hbpacker.num_recycles = 3

    config.model.hbpacker.pack_crop = 10.0
    config.model.hbpacker.knn_k = 24

    return config


CONFIGS = {
    "hbpacker_longleaf": build_hbpacker_config_longleaf,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hbpacker_longleaf")
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
    trainer = HBPackerTrainer(config)

    # Load checkpoint if resuming
    if args.resume_ckpt:
        assert os.path.isfile(args.resume_ckpt), (
            f"Invalid checkpoint file {args.resume_ckpt} specified."
        )
    # Set up wandb.
    if args.use_wandb:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        if not args.resume_id:
            wandb.init(project="HBPacker", config=OmegaConf.to_container(trainer.cfg))
        else:
            wandb.init(
                project="HBPacker",
                config=OmegaConf.to_container(trainer.cfg),
                id=args.resume_id,
                resume="must",
            )

    # Fit HBPacker
    trainer.fit()
