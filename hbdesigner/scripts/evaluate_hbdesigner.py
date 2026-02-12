import argparse
import os
import time
from pathlib import Path
import numpy as np
from hbdesigner.data.hbnet import initialize_rosetta
from hbdesigner.scripts.train_hbdesigner import HBDesignerTrainer
from hbdesigner.scripts.train_hbpacker import HBPackerTrainer
from hbdesigner.utils import get_config_from_file, seed_everything

if __name__ == "__main__":
    # This script is meant to gather validation metrics from the FULL test set for HBDesigner.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design_config",
        type=str,
        default="dev",
        help="Model config file.",
    )
    parser.add_argument(
        "--design_ckpt",
        type=str,
        default="",
        help="Model ckpt file. Not needed for 'test_loop_native.",
    )
    parser.add_argument(
        "--pack_config",
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
        default="hbpacker",
        choices=["rosetta", "hbpacker", "pippack"],
        help="Pack method. Options are (rosetta, hbpacker, pippack).",
    )
    parser.add_argument(
        "--pack_mode",
        type=str,
        default="fast",
        choices=["fast", "slow"],
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
        help="Collect and score the first N samples for benchmarking. Off by default. Overrides --n_batches if enabled.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats for error bars. Defaults to 1.",
    )
    args = parser.parse_args()
    print("Args:", args)

    # 1. Set up design config/model
    print("Initializing design model...")
    # General params
    design_config = get_config_from_file(args.design_config)
    design_config.seed = 1234
    seed_everything(design_config.seed)
    design_config.log_dir = None
    # Use workers for packing instead
    design_config.num_workers = 0

    # Data sampling params
    design_config.model.hbdesigner.min_res = 2
    design_config.model.hbdesigner.max_res = 6
    design_config.model.hbdesigner.inter_weight = 1.0
    design_config.model.hbdesigner.hbnet_pct = 0.0
    design_config.model.hbdesigner.rescore = False
    design_config.model.hbdesigner.rescore_filter = False

    # Conditioning info params
    design_config.model.hbdesigner.guide_atom_pct = 0.0
    design_config.model.hbdesigner.guide_atom_sigma = 4.0
    design_config.model.hbdesigner.seq_cond_pct = 0.0
    design_config.model.hbdesigner.seq_cond_unk_pct = 0.0

    # Eval params
    design_config.model.hbdesigner.bb_noise = 0.0
    design_config.model.hbdesigner.batch_size = args.batch_size

    steps = args.n_batches
    if args.first_n is not None:
        steps = None

    design_trainer = HBDesignerTrainer(design_config, print_config=False)
    design_trainer.load_model_state(args.design_ckpt)

    # 2. Set up packing config/model (won't be using dataloaders)
    print("Initializing packing model...")
    # General params
    pack_config = get_config_from_file(args.pack_config)
    pack_config.seed = 1234
    seed_everything(pack_config.seed)
    pack_config.log_dir = None

    # Rosetta pack/min uses mp, so we zero this out here to avoid multiplication of workers
    pack_config.num_workers = 0

    # Data sampling params here
    pack_config.model.hbpacker.min_res = 2
    pack_config.model.hbpacker.max_res = 6
    pack_config.model.hbpacker.bb_noise = 0.0

    pack_config.model.pippack.ckpt = os.path.join(Path(__file__).parents[2], "model_weights/pippack_model_1_ckpt.pt")

    pack_config.model.hbpacker.pack_method = args.pack_method
    pack_config.model.hbpacker.pack_mode = args.pack_mode
    pack_config.model.hbpacker.pack_min = args.pack_min

    pack_trainer = HBPackerTrainer(pack_config, print_config=False)
    pack_trainer.load_model_state(args.pack_ckpt)
    initialize_rosetta(args.pack_mode)

    # 3. Run design/packing loop
    repeat_info = {}
    seeds = [1234, 1111, 42, 10124, 4529]
    for r in range(args.repeats):
        t0 = time.time()
        seed_everything(seeds[r])
        test_dl = design_trainer.build_test_data_loader()
        test_dl_pack = pack_trainer.build_test_data_loader()
        print(f"Starting test loop # {r + 1} / {args.repeats}")

        info = design_trainer.test_loop(
            test_dl,
            steps=steps,
            n_workers=args.num_workers,
            dump=args.dump,
            first_n=args.first_n,
            seq_sample_temp=args.seq_temp,
            res_sample_temp=args.pos_temp,
            verbose=False,
            pack_trainer=pack_trainer,
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
    try:
        print(f"{key}: {np.mean(value)} +- {np.std(value)}")
    except TypeError:
        print(f"{key}: None")
