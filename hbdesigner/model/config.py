from dataclasses import dataclass, field

from hbdesigner.utils import StrictDataClass


@dataclass
class HBPackerModelConfig(StrictDataClass):
    """HBPacker model configuration.

    Attributes:
        pn_dim (int): Hidden dimension size for protein nodes. Defaults to 128.
        pe_dim (int): Hidden dimension size for protein edges. Defaults to 128.
        mlp_inner_dim (int): Hidden dimension for inner layers of FFWD layer MLPs. Defaults to 128.
        knn_k (int): Number of nearest neighbors to use in protein graph. Defaults to 30.
        num_protein_encoder_layers (int): Number of MPNN layers in ProteinEncoder. Defaults to 3.
        max_seq_sep (int): Maximum separation in sequence space for positional
            encodings. Defaults to 32.
        num_rbf (int): Number of RBF encodings to use. Defaults to 16.
        num_mlp_layers (int): Number of MLP layers. Defaults to 3.
        bb_noise (float): Backbone coordinate noise added during training (in A). Defaults to 0.0.
        chi_mse_weight (float): Weight for chi angle MSE regression loss. Defaults to 1.0.
        chi_norm_weight (float): Weight for chi angle normalization penalty. Defaults to 0.01.
        sc_msd_weight (float): Weight for sidechain xyz MSD loss. Defaults to 1.0.
        sc_clash_weight (float): Weight for sidechain xyz VdW clash loss. Defaults to 0.0.
        orient_msd_weight (float): Weight for sidechain xyz relative orientation loss. Defaults to 0.0.
        reweight_chi_mse (bool): Whether to reweight chi MSE loss by relative chi freq. Defaults to False.
        num_recycles (int): Number of packing recycles. Defaults to 0.
        data_location (str): Path to training dataset. Defaults to "".
        inter_weight (float): Sampling weight for interface samples relative to intra-chain samples. Defaults to 1.0.
        hbnet_pct (float): Sampling pct for using hbnet samples, if available. Defaults to 0.0.
        max_res (int): Maximum size network to consider, inclusive. Defaults to 6.
        min_res (int): Minimum size network to consider, inclusive. Defaults to 2.
        pack_method (str): Which packer to use for packing when sampling. Defaults to "none". Options are ("none", "rosetta", "hbdes3").
        pack_mode (str): What packing 'mode' to use when packing. Defaults to "fast". Options are ("fast", "slow").
        pack_min (bool): Whether to run Rosetta MinMover on packing outputs. Defaults to True.
        pack_crop (float): Threshold for cropping before packing, in Angstrom. Defaults to -1 (no crop).
    """

    pn_dim: int = 128
    pe_dim: int = 128
    mlp_inner_dim: int = 128
    knn_k: int = 30
    num_protein_encoder_layers: int = 3
    max_seq_sep: int = 32
    num_rbf: int = 16
    num_mlp_layers: int = 3

    # Loss terms
    chi_mse_weight: float = 1.0
    chi_norm_weight: float = 0.01
    sc_msd_weight: float = 1.0
    sc_clash_weight: float = 0.0
    orient_msd_weight: float = 0.0
    reweight_chi_mse: bool = False
    num_recycles: int = 0

    # Data configuration
    data_location: str = ""
    inter_weight: float = 1.0
    hbnet_pct: float = 0.0
    batch_size: int = 10_000
    bb_noise: float = 0.0
    max_res: int = 6
    min_res: int = 2
    rescore: bool = False
    rescore_filter: bool = False

    # Packer configuration
    pack_method: str = "none"
    pack_mode: str = "fast"
    pack_min: bool = True
    pack_crop: float = -1.


@dataclass
class HBDesignerModelConfig(StrictDataClass):
    """HBDesigner model configuration.

    Attributes:
        pn_dim (int): Hidden dimension size for protein nodes. Defaults to 128.
        pe_dim (int): Hidden dimension size for protein edges. Defaults to 128.
        pg_dim (int): Hidden dimension size for protein graph nodes. Defaults to 256.
        mlp_inner_dim (int): Hidden dimesion for inner layers of FFWD layer MLPs. Defaults to 128.
        knn_k (int): Number of nearest neighbors to use in protein graph. Defaults to 30.
        num_protein_encoder_layers (int): Number of MPNN layers in ProteinEncoder. Defaults to 3.
        max_seq_sep (int): Maximum separation in sequence space for positional
            encodings. Defaults to 32.
        num_rbf (int): Number of RBF encodings to use. Defaults to 16.
        num_mlp_layers (int): Number of MLP layers. Defaults to 3.
        bb_noise (float): Backbone coordinate noise added during training (in A). Defaults to 0.0.
        net_res_nll_weight (float): Weight for the residue prediction NLL
            loss. Defaults to 1.0.
        seq_nll_weight (float): Weight for the sequence prediction NLL loss.
            Defaults to 1.0.
        loss_type (str): Which loss formula to use. Defaults to "nll". Options are ("nll", "focal").
        focal_gamma (float): Gamma hyperparameter for focal loss. Defaults to 0.0. Higher means stronger modulation away from CE loss.
        nll_smoothing (float): Weight for NLL label smoothing. Defaults to 0.0.
        data_location (str): Path to training dataset. Defaults to "".
        inter_weight (float): Sampling weight for interface samples relative to intra-chain samples. Defaults to 1.0.
        hbnet_pct (float): Sampling pct for using hbnet samples, if available. Defaults to 0.0.
        max_res (int): Maximum size network to consider, inclusive. Defaults to 6.
        min_res (int): Minimum size network to consider, inclusive. Defaults to 2.
        guide_atom_pct (float): Percent of samples to use guide atom conditioning. Default is 0.0.
        guide_atom_sigma (float): Sigma of normal distribution for guide atom placement. Default is 1.0.
        seq_cond_pct (float): Percent of samples to use expected seq conditioning. Default is 0.0.
        seq_cond_unk_pct (float): Percent of samples to partially mask for seq conditioning. Default is 0.0.
    """

    # Model architecture
    pn_dim: int = 128
    pe_dim: int = 128
    pg_dim: int = 256
    mlp_inner_dim: int = 128
    knn_k: int = 30
    num_protein_encoder_layers: int = 3
    max_seq_sep: int = 32
    num_rbf: int = 16
    num_mlp_layers: int = 3

    # Loss weights
    net_res_nll_weight: float = 1.0
    seq_nll_weight: float = 2.0
    nll_smoothing: float = 0.0
    loss_type: str = "nll"
    focal_gamma: float = 0.0

    # Data configuration
    data_location: str = ""
    inter_weight: float = 1.0
    hbnet_pct: float = 0.0
    batch_size: int = 10_000
    max_res: int = 6
    min_res: int = 2
    bb_noise: float = 0.0
    rescore: bool = False
    rescore_filter: bool = False

    # Conditioning info
    guide_atom_pct: float = 0.0
    guide_atom_sigma: float = 1.0
    seq_cond_pct: float = 0.0
    seq_cond_unk_pct: float = 0.0


@dataclass
class PIPPackModelConfig(StrictDataClass):
    """PIPPack model configuration.

    Attributes:
        ckpt (str): Path to the PIPPack checkpoint to use. Defaults to "".
        use_ipmp (bool): Whether to use IPMP or MPNN layers in PIPPack.
            Defaults to True.
        n_points (int): Number of invariant points to use in IPMP in PIPPack.
            Defaults to 8.
        recycle_SC_D_sc (bool): Enables recycling predicted sidechain
            dihedral information as sin/cos encoding in PIPPack. Defaults to True.
        mask_distances (bool): Enables masking of distances of atoms that may
            be nonexistant. Defaults to True.
        device (str): Which device to use for PIPPack hosting. Defaults to "cuda".
        recycles (int): How many recycles to run for PIPPack. Defaults to 0.
        resampling (bool): Whether to use PIPPack resampling for postprocessing. Defaults to False.
    """

    # PIPPack model parameters
    # These are non-default values from pippack_model_1_config.pickle
    ckpt: str = ""
    use_ipmp: bool = True
    n_points: int = 8
    recycle_SC_D_sc: bool = True
    mask_distances: bool = True
    device: str = "cuda"
    recycles: int = 0
    resampling: bool = False


@dataclass
class ModelConfig(StrictDataClass):
    """Master configuration for models.

    Attributes:
        model_name (str): The name of the model to use. Defaults to "Uniform".
        dropout (float): Dropout rate. Defaults to 0.1.
        frankenpacker (FrankenPackerModelConfig): The FrankenPacker model
            configuration.
    """

    model_name: str = "Uniform"
    dropout: float = 0.1
    pippack: PIPPackModelConfig = field(
        default_factory=PIPPackModelConfig
    )
    hbdesigner: HBDesignerModelConfig = field(default_factory=HBDesignerModelConfig)
    hbpacker: HBPackerModelConfig = field(default_factory=HBPackerModelConfig)
