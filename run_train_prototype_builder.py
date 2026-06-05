# run for train_prototype_builder.py
# run command: uv run python run_train_prototype_builder.py --config "src/configs/cfg_prototype_builder.yaml"

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engines.train_prototype_builder import train


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training Config")

    p.add_argument('--config', type=str, required=True, help='config file path')
    # # for debug
    # p.add_argument('--config', default='src/configs/cfg_prototype_builder.yaml', type=str, help='config file path')

    return p.parse_args()


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    train(args.config)


if __name__ == "__main__":
    main()
