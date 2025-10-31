from dataclasses import dataclass, field

from proteingfn.utils import StrictDataClass


@dataclass
class HBDesignerModelConfig(StrictDataClass):
    """HBDesigner model configuration.

    Attributes:
        pn_dim (int): Hidden dimension size for protein nodes. Defaults to 128.
        pe_dim (int): Hidden dimension size for protein edges. Defaults to 128.
        pg_dim (int): Hidden dimension size for protein graph nodes. Defaults to 256.
        gn_dim (int): Hidden dimension size for gig nodes. Defaults to 64.
        ge_dim (int): Hidden dimension size for gig edges. Defaults to 64.
        gg_dim (int): Hidden dimension size for gig graph nodes. Defaults to 128.
        mlp_inner_dim (int): Hidden dimesion for inner layers of FFWD layer MLPs. Defaults to 128.
        knn_k (int): Number of nearest neighbors to use in protein graph. Defaults to 30.
        num_protein_encoder_layers (int): Number of MPNN layers in ProteinEncoder. Defaults to 3.
        num_protein_decoder_layers (int): Number of MPNN layers in ProteinDecoder. Defaults to 3.
        max_seq_sep (int): Maximum separation in sequence space for positional
            encodings. Defaults to 32.
        num_rbf (int): Number of RBF encodings to use. Defaults to 16.
        num_mpnn_layers (int): Number of MPNN layers. Defaults to 3.
        num_mlp_layers (int): Number of MLP layers. Defaults to 3.
        num_ipmp_points (int): Number of invariant points to use in IPMP. Defaults to 8.
        use_ipmp (bool): Use IPMP instead of MPNN in message passing. Defaults to False.
        use_graph_comms (bool): Use GraphCommunications layers in Encoder. Defaults to False.
        use_bb_dih (bool): Use bb dihedral sin/cos as node features. Defaults to False.
        bb_noise (float): Backbone coordinate noise added during training (in A). Defaults to 0.0.
        net_res_nll_weight (float): Weight for the HBNet residue prediction NLL
            loss. Defaults to 1.0.
        seq_nll_weight (float): Weight for the sequence prediction NLL loss.
            Defaults to 1.0.
        edge_nll_weight (float): Weight for the HBNet edge prediction NLL loss.
            Defaults to 1.0.
        chi_mse_weight (float): Weight for chi angle MSE regression loss. Defaults to 1.0.
        chi_norm_weight (float): Weight for chi angle normalization penalty. Defaults to 0.01.
        sc_msd_weight (float): Weight for sidechain xyz MSD loss. Defaults to 1.0.
        sc_clash_weight (float): Weight for sidechain xyz VdW clash loss. Defaults to 0.0.
        orient_msd_weight (float): Weight for sidechain xyz relative orientation loss. Defaults to 0.0.
        reweight_chi_mse (bool): Whether to reweight chi MSE loss by relative chi freq. Defaults to False.
        num_recycles (int): Number of packing recycles. Defaults to 0.
        pack (bool): Whether to use packing model. Defaults to False.
        grad_ckpt (bool): Whether to use gradient checkpointing during training. Defaults to False.
        loss_norm (float): Empirical loss normalization factor. Defaults to 2000.
        loss_type (str): Which loss formula to use. Defaults to "nll". Options are ("nll", "focal").
        focal_gamma (float): Gamma hyperparameter for focal loss. Defaults to 0.0. Higher means stronger modulation away from CE loss.
        nll_smoothing (float): Weight for NLL label smoothing. Defaults to 0.0.
        decoding_scheme (str): Decoding mask to use for training. Defaults to "default". Options are ("default", "autoreg").
        data_location (str): Path to training dataset. Defaults to "".
        data_subset (float): Fraction of dataset to use. Defaults to 1.0.
        keep_Hs (bool): Whether to retain H atoms on data loading. Defaults to False. Only used for native-repacking.
        crop_method (str): Method of cropping to use, if crop enabled. Defaults to "random". Options are ("random", "centered")
        inter_weight (float): Sampling weight for interface samples relative to intra-chain samples. Defaults to 1.0.
        hbnet_pct (float): Sampling pct for using hbnet samples, if available. Defaults to 0.0.
        max_res (int): Maximum size network to consider, inclusive. Defaults to 6.
        min_res (int): Minimum size network to consider, inclusive. Defaults to 2.
        pack_method (str): Which packer to use for packing when sampling. Defaults to "none". Options are ("none", "rosetta", "hbdes3").
        pack_mode (str): What packing 'mode' to use when packing. Defaults to "fast". Options are ("fast", "slow").
        pack_min (bool): Whether to run Rosetta MinMover on packing outputs. Defaults to True.
        pack_crop (float): Threshold for cropping before packing, in Angstrom. Defaults to -1 (no crop).
        pred_nodes (bool): Predict node labels (hbonded/not-hbonded) in addition to sequence. Default is False.
        pred_edges (bool): Predict edge labels (bonding/non-bonding) in addition to sequence. Default is False.
        guide_atom_pct (float): Percent of samples to use guide atom conditioning. Default is 0.0.
        guide_atom_sigma (float): Sigma of normal distribution for guide atom placement. Default is 1.0.
        seq_cond_pct (float): Percent of samples to use expected seq conditioning. Default is 0.0.
        seq_cond_unk_pct (float): Percent of samples to partially mask for seq conditioning. Default is 0.0.
        min_res_decoded (int): Minimum number of residues already decoded in training/eval examples. Default is 0.
    """

    pn_dim: int = 128
    pe_dim: int = 128
    pg_dim: int = 256
    gn_dim: int = 64
    ge_dim: int = 64
    gg_dim: int = 128
    mlp_inner_dim: int = 128
    knn_k: int = 30
    num_protein_encoder_layers: int = 3
    num_protein_decoder_layers: int = 3
    max_seq_sep: int = 32
    num_rbf: int = 16
    num_mlp_layers: int = 3
    num_ipmp_points: int = 8
    use_ipmp: bool = False
    use_graph_comms: bool = False
    use_bb_dih: bool = False
    bb_noise: float = 0.0

    # Loss weights
    net_res_nll_weight: float = 1.0
    seq_nll_weight: float = 2.0
    edge_nll_weight: float = 1.0

    # Packing terms
    pack: bool = False
    chi_mse_weight: float = 1.0
    chi_norm_weight: float = 0.01
    sc_msd_weight: float = 1.0
    sc_clash_weight: float = 0.0
    orient_msd_weight: float = 0.0
    reweight_chi_mse: bool = False
    num_recycles: int = 0

    decoding_scheme: str = "default"
    nll_smoothing: float = 0.0
    grad_ckpt: bool = False
    loss_norm: float = 2000.0
    loss_type: str = "nll"
    focal_gamma: float = 0.0

    # Data configuration
    data_location: str = ""
    data_subset: float = 1.0
    keep_Hs: bool = False
    crop_method: str = "random"
    inter_weight: float = 1.0
    hbnet_pct: float = 0.0
    batch_size: int = 10_000
    max_res: int = 6
    min_res: int = 2
    rescore: bool = False
    rescore_filter: bool = False

    # Packer configuration
    pack_method: str = "none"
    pack_mode: str = "fast"
    pack_min: bool = True
    pack_crop: float = -1.

    # Node/edge prediction
    pred_nodes: bool = False
    pred_edges: bool = False

    # Conditioning info parameters
    guide_atom_pct: float = 0.0
    guide_atom_sigma: float = 1.0
    min_res_decoded: int = 0
    seq_cond_pct: float = 0.0
    seq_cond_unk_pct: float = 0.0


@dataclass
class FrankenPackerModelConfig(StrictDataClass):
    """FrankenPacker model configuration.

    Attributes:
        pippack_ckpt (str): Path to the PIPPack checkpoint to use. Defaults to "".
        pippack_use_ipmp (bool): Whether to use IPMP or MPNN layers in PIPPack.
            Defaults to True.
        pippack_n_points (int): Number of invariant points to use in IPMP in PIPPack.
            Defaults to 8.
        pippack_recycle_SC_D_sc (bool): Enables recycling predicted sidechain
            dihedral information as sin/cos encoding in PIPPack. Defaults to True.
        pippack_mask_distances (bool): Enables masking of distances of atoms that may
            be nonexistant. Defaults to True.
        pippack_device (str): Which device to use for PIPPack hosting. Defaults to "cuda".
        pippack_recycles (int): How many recycles to run for PIPPack. Defaults to 0.
        pippack_resampling (bool): Whether to use PIPPack resampling for postprocessing. Defaults to False.
        ligandmpnn_ckpt (str): Path to the LigandMPNN checkpoint to use. Defaults to "".
        ligandmpnn_k_neighbors (int): Number of neighbors for each residue in
            LigandMPNN. Defaults to 32.
        ligandmpnn_atom_context_num (int): Number of atoms to use as ligand context in
            LigandMPNN. Defaults to 25.
        ligandmpnn_model_type (str): Which model type to use for ProteinMPNN class.
            Defaults to "ligand_mpnn".
        ligandmpnn_use_side_chain_context (bool): Whether to use side chain atoms as
            ligand context in LigandMPNN. Defaults to True.
        no_scaling (bool): Disables the scale module and uses a scale of 1.0. Defaults
            to False.
        pn_dim (int): Hidden dimension size for protein nodes. Defaults to 128.
        pe_dim (int): Hidden dimension size for protein edges. Defaults to 128.
        pg_dim (int): Hidden dimension size for protein graph nodes. Defaults to 256.
        knn_k (int): Number of nearest neighbors to use in protein graph. Defaults to 30.
        max_seq_sep (int): Maximum separation in sequence space for positional
            encodings. Defaults to 32.
        num_rbf (int): Number of RBF encodings to use. Defaults to 16.
        num_mpnn_layers (int): Number of MPNN layers. Defaults to 3.
        num_mlp_layers (int): Number of MLP layers. Defaults to 3.
        num_ipmp_points (int): Number of invariant points to use in IPMP. Defaults to 8.

        logZ_* (Union[str, int]): Hyperparameters associated with the logZ parameter.
            See descriptions above.
    """

    # PIPPack model parameters
    # These are non-default values from pippack_model_1_config.pickle
    pippack_ckpt: str = ""
    pippack_use_ipmp: bool = True
    pippack_n_points: int = 8
    pippack_recycle_SC_D_sc: bool = True
    pippack_mask_distances: bool = True
    pippack_device: str = "cuda"
    pippack_recycles: int = 0
    pippack_resampling: bool = False

    # LigandMPNN model parameters
    # These are non-default values from ligandmpnn_v_32_010_25.pt
    ligandmpnn_ckpt: str = ""
    ligandmpnn_k_neighbors: int = 32
    ligandmpnn_atom_context_num: int = 25
    ligandmpnn_model_type: str = "ligand_mpnn"
    ligandmpnn_use_side_chain_context: bool = True

    # Scale module parameters
    no_scaling: bool = False
    pn_dim: int = 128
    pe_dim: int = 128
    pg_dim: int = 256
    knn_k: int = 30
    max_seq_sep: int = 32
    num_rbf: int = 16
    num_mpnn_layers: int = 3
    num_mlp_layers: int = 3
    num_ipmp_points: int = 8

    # LogZ parameters
    logZ_pn_dim: int = 64
    logZ_pe_dim: int = 64
    logZ_pg_dim: int = 128
    logZ_knn_k: int = 30
    logZ_max_seq_sep: int = 32
    logZ_num_rbf: int = 16
    logZ_num_mpnn_layers: int = 3
    logZ_num_mlp_layers: int = 3
    logZ_num_ipmp_points: int = 8


@dataclass
class ModelConfig(StrictDataClass):
    """Master configuration for models.

    Attributes:
        model_name (str): The name of the model to use. Defaults to "Uniform".
        dropout (float): Dropout rate. Defaults to 0.1.
        gig_matcher (GIGMatcherModelConfig): The GIGMatcher model
            configuration.
        hbnet (HBNetModelConfig): The HBNet model configuration.
        seq_design (SeqDesignModelConfig): The SeqDesign model
            configuration.
        frankenpacker (FrankenPackerModelConfig): The FrankenPacker model
            configuration.
        gigpacker (GIGPackerModelConfig): The GIGPacker model configuration.
        gigdesigner (GIGDesignerModelConfig): The GIGDesigner model configuration.
    """

    model_name: str = "Uniform"
    dropout: float = 0.1
    gig_matcher: GIGMatcherModelConfig = field(default_factory=GIGMatcherModelConfig)
    hbnet: HBNetModelConfig = field(default_factory=HBNetModelConfig)
    frankenpacker: FrankenPackerModelConfig = field(
        default_factory=FrankenPackerModelConfig
    )
    seq_design: SeqDesignModelConfig = field(default_factory=SeqDesignModelConfig)
    gigpacker: GIGPackerModelConfig = field(default_factory=GIGPackerModelConfig)
    gigdesigner: GIGDesignerModelConfig = field(default_factory=GIGDesignerModelConfig)
    hbdesigner: HBDesignerModelConfig = field(default_factory=HBDesignerModelConfig)
