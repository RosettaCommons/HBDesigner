import argparse
import time
import os
import numpy as np
from pathlib import Path

from hbdesigner.data.hbnet import initialize_rosetta
from hbdesigner.scripts.train_hbpacker import (
    HBPackerTrainer,
)
from hbdesigner.utils import seed_everything, get_config_from_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluation script for HBPacker and other packing methods."
    )
    parser.add_argument(
        "--pack_config",
        type=str,
        required=True,
        help="Model config file.",
    )
    parser.add_argument(
        "--pack_ckpt",
        type=str,
        required=True,
        help="Model ckpt file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for packing/dataloading. Default is 1.",
    )
    parser.add_argument(
        "--pack_method",
        type=str,
        default="hbpacker",
        choices=["rosetta", "hbpacker", "pippack", "native"],
        help="Pack method. Options are (rosetta, hbpacker, pippack, native).",
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
        help="Whether to minimize after packing. Defaults to False.",
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

    # Set up HBDesigner config
    config = get_config_from_file(args.pack_config)
    config.seed = 1234
    seed_everything(config.seed)
    config.log_dir = None

    # Rosetta pack/min uses mp, so we zero this out here to avoid multiplication of workers
    config.num_workers = 0

    # Data sampling params here
    config.model.hbpacker.min_res = 2
    config.model.hbpacker.max_res = 6
    config.model.hbpacker.inter_weight = 1.0
    config.model.hbpacker.hbnet_pct = 0.0
    config.model.hbpacker.rescore = False
    config.model.hbpacker.rescore_filter = False
    config.model.hbpacker.bb_noise = 0.0

    steps = args.n_batches
    config.model.hbpacker.batch_size = args.batch_size

    config.model.pippack.ckpt = os.path.join(Path(__file__).parents[2], "model_weights/pippack_model_1_ckpt.pt")

    config.model.hbpacker.pack_method = args.pack_method
    config.model.hbpacker.pack_mode = args.pack_mode
    config.model.hbpacker.pack_min = args.pack_min

    # Set up benchmarking params
    if args.first_n is not None:
        steps = None

    trainer = HBPackerTrainer(config, print_config=False)
    trainer.load_model_state(args.pack_ckpt)
    initialize_rosetta(args.pack_mode)

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
            steps=steps,
            n_workers=args.num_workers,
            dump=args.dump,
            first_n=args.first_n,
            verbose=False,
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

print("Meta evaluation w/repeats:")
for key, value in repeat_info.items():
    try:
        print(f"{key}: {np.mean(value)} +- {np.std(value)}")
    except TypeError:
        print(f"{key}: None")
