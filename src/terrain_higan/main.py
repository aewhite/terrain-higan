import argparse

import torch

from terrain_higan.projects.z0_generator.train import train_z0_generator
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.train import train_zn_map_upscaler
from terrain_higan.tensorboard_utils import tensorboard_helpers


def setup_pytorch():
    torch.autograd.set_detect_anomaly(False)

    torch.set_printoptions(
        linewidth=120,    # characters per line before wrapping
        edgeitems=5,      # number of array elements to show at beginning/end of array
        threshold=1000,   # how many elements before summarizing with "..."
        sci_mode=False
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train the map-upscaler chain")
    parser.add_argument("--tensorboard-host")
    parser.add_argument("--tensorboard-port", type=int)
    args = parser.parse_args(argv)
    setup_pytorch()
    tensorboard_helpers.start_tensorboard_detached(host=args.tensorboard_host, port=args.tensorboard_port)
    # train_zn_map_upscaler(ZnUpscalerConfig(z_level=1, profile=True, profile_only=True))
    # train_z0_generator()
    for z in range(1, 21):
        train_zn_map_upscaler(ZnUpscalerConfig(z_level=z))

if __name__ == "__main__":
    main()
