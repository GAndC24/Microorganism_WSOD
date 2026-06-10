# run for eval_prototype.py
# run command: uv run python run_eval_prototype.py --config "src/configs/cfg_eval_prototype.yaml"

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engines.eval_prototype import eval


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training Config")

    p.add_argument('--config', type=str, required=True, help='config file path')
    # # for debug
    # p.add_argument('--config', default='src/configs/cfg_eval_prototype.yaml', type=str, help='config file path')

    return p.parse_args()


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    eval(args.config)


if __name__ == "__main__":
    main()
