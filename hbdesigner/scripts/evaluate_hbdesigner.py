import argparse
import time
import numpy as np

from hbdesigner.data.hbnet import initialize_rosetta
from hbdesigner.scripts.train_hbdesigner import (
    HBDesignerTrainer,
    get_config_from_file,
)
from hbdesigner.model.hbdesign_model import HBDesigner3
from hbdesigner.utils import seed_everything


if __name__ == "__main__":
    # This script is meant to gather validation metrics from the FULL test set for HBDesigner.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="dev",
        help="Model config file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Model ckpt file. Not needed for 'test_loop_native.",
    )
    parser.add_argument(
        "--design",
        action="store_true",
        help="Whether to use HBDes3 seq design. Defaults to False (uses native seq).",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Whether to use HBDes3/Rosetta packing. Defaults to False.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for packing/dataloading. Default is 1.",
    )
    parser.add_argument(
        "--seq_temp",
        type=float,
        default=0.1,
        help="Sequence sampling temperature (lower is more strict). Default is 0.1.",
    )
    parser.add_argument(
        "--pos_temp",
        type=float,
        default=0.1,
        help="Sequence sampling temperature (lower is more strict). Default is 0.1.",
    )
    parser.add_argument(
        "--pack_method",
        type=str,
        default="rosetta",
        help="Pack method. Options are (rosetta, hbdes3, pippack, native).",
    )
    parser.add_argument(
        "--pack_mode",
        type=str,
        default="fast",
        help="Pack mode. Options are (fast, slow).",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=10,
        help="Number of batches to process. Default is 10.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1_000,
        help="Batch size for eval. Default is 1000 tokens.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Whether to dump PDBs for inspection. Defaults to False.",
    )
    parser.add_argument(
        "--pack_min",
        action="store_true",
        help="Whether to minimize for packing. Defaults to False.",
    )
    parser.add_argument(
        "--first_n", 
        type=int, 
        default=None,
        help="Collect and score the first N samples for benchmarking. Off by default. Overrides --n_batches if enabled."
    )
    parser.add_argument(
        "--repeats", 
        type=int, 
        default=1,
        help="Number of repeats for error bars. Defaults to 1."
    )
    parser.add_argument(
        "--pack_config_file",
        type=str,
        default="dev",
        help="Model config file.",
    )
    parser.add_argument(
        "--pack_ckpt",
        type=str,
        default="",
        help="Model ckpt file. Not needed for 'test_loop_native.",
    )
    args = parser.parse_args()
    config = get_config_from_file(args.config_file)

    # Set up the trainer.
    config.seed = 1234
    seed_everything(config.seed)
    config.log_dir = None
    config.num_workers = args.num_workers

    if not args.pack:
        args.pack_method = "native"

    config.model.hbdesigner.pack_method = args.pack_method
    config.model.hbdesigner.pack_mode = args.pack_mode
    config.model.hbdesigner.pack_min = args.pack_min
    config.model.hbdesigner.num_recycles = 3
    config.model.hbdesigner.pack_crop = -1
    config.model.hbdesigner.min_res = 2
    config.model.hbdesigner.max_res = 6

    # NOTE: key inference params here
    config.model.hbdesigner.guide_atom_pct = 0.0
    config.model.hbdesigner.guide_atom_sigma = 4.0
    config.model.hbdesigner.seq_cond_pct = 0.0
    config.model.hbdesigner.seq_cond_unk_pct = 0.0
    config.model.hbdesigner.inter_weight = 1.0
    config.model.hbdesigner.hbnet_pct = 0.0

    print(config.model.hbdesigner.pack_method, '***', config.model.hbdesigner.pack_mode, config.model.hbdesigner.pack_min)
    steps = args.n_batches
    config.model.hbdesigner.batch_size = args.batch_size
    config.model.frankenpacker.pippack_ckpt = "/users/d/i/dieckhau/dev/ProteinGFN/proteingfn/data/model_weights/pippack_model_1_ckpt.pt"
    # Rosetta packing requires multiproc, but dataloading doesn't
    config.num_workers = 0

    config.model.hbdesigner.rescore = False
    config.model.hbdesigner.rescore_filter = False

    # Only score first N networks for benchmarking
    if args.first_n is not None:
        steps = None

    trainer = HBDesignerTrainer(config, print_config=False)
    initialize_rosetta(args.pack_mode)
    print(f"Sampling temperatures (seq/pos): {args.seq_temp}/{args.pos_temp}")

    # NOTE on different test loop options
    # - design=True, pack=True: design with model, pack with packer - full eval
    # - design=False, pack=True: keep native seq, pack with packer - packing eval
    # - design=True, pack=False: INVALID - not allowed
    # - design=False, pack=False, keep native seq, keep native pack - scoring eval
    if args.design and (not args.pack):
        raise ValueError("Can't design but not pack!")

    if args.design:
        trainer.load_model_state(args.ckpt, packer=False)
    if args.pack:
        pack_config = get_config_from_file(args.pack_config_file)
        # Set up the trainer.
        # pack_config.seed = 1234
        # seed_everything(pack_config.seed)
        pack_config.log_dir = None
        pack_config.num_workers = args.num_workers

        pack_config.model.hbdesigner.pack_method = args.pack_method
        pack_config.model.hbdesigner.pack_mode = args.pack_mode
        pack_config.model.hbdesigner.pack_min = args.pack_min
        pack_config.model.hbdesigner.num_recycles = 3
        pack_config.model.hbdesigner.pack_crop = -1
        pack_config.model.hbdesigner.min_res = 2
        pack_config.model.hbdesigner.max_res = 6

        # NOTE: key inference params here
        pack_config.model.hbdesigner.guide_atom_pct = 0.0
        pack_config.model.hbdesigner.guide_atom_sigma = 4.0
        pack_config.model.hbdesigner.seq_cond_pct = 0.0
        pack_config.model.hbdesigner.seq_cond_unk_pct = 0.0
        pack_config.model.hbdesigner.inter_weight = 1.0
        pack_config.model.hbdesigner.hbnet_pct = 0.0

        pack_config.model.hbdesigner.batch_size = args.batch_size
        pack_config.model.frankenpacker.pippack_ckpt = "/users/d/i/dieckhau/dev/ProteinGFN/proteingfn/data/model_weights/pippack_model_1_ckpt.pt"
        # Rosetta packing requires multiproc, but dataloading doesn't
        pack_config.num_workers = 0

        trainer.pack_model = HBDesigner3(pack_config)
        trainer.pack_model.to(pack_config.device)
        trainer.load_model_state(args.pack_ckpt, packer=True)

    # Enable repeated runs for errorbars
    repeat_info = {}
    seeds = [1234, 1111, 42, 10124, 4529]
    for r in range(args.repeats):
        t0 = time.time()
        seed_everything(seeds[r])
        test_dl = trainer.build_test_data_loader()
        print(f"Starting test loop # {r + 1} / {args.repeats}")

        info = trainer.test_loop(
            test_dl,
            design=args.design,
            pack=args.pack,
            steps=steps,
            n_workers=args.num_workers,
            dump=args.dump,
            first_n=args.first_n,
            seq_sample_temp=args.seq_temp,
            res_sample_temp=args.pos_temp,
            verbose=True,
)
        t1 = time.time()
        elapsed = round(t1 - t0)
        for key, value in info.items():

            if key not in repeat_info:
                repeat_info[key] = [value]
            else:
                repeat_info[key].append(value)
        print(f"Finished test loop # {r + 1} / {args.repeats}")
        print(f"Runtime: {elapsed}")
        print("." * 50)


print("Meta eval w/repeats:")
for key, value in repeat_info.items():
    print(f"{key}: {np.mean(value)} +- {np.std(value)}")
