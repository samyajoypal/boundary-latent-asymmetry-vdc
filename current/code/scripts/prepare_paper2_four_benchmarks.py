"""Prepare the fixed four benchmark inputs without choosing by observed results."""
import argparse
import json

from ghmvae.data.publication_benchmarks import EXPECTED, prepare_image_features, prepare_reuters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=EXPECTED)
    parser.add_argument("--data-home", default="data/paper2_four_benchmarks")
    parser.add_argument("--image-home", default="data/torchvision")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.dataset == "reuters10k":
        metadata = prepare_reuters(args.data_home)
    else:
        metadata = prepare_image_features(args.data_home, args.dataset, args.image_home, args.device)
    print(json.dumps(metadata, indent=2))
