import argparse

from classifier.code.data import generate_nicopp_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/nicopp")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--metadata_prefix", type=str, default="metadata")
    parser.add_argument("--split_seed", "--seed", type=int, default=42)
    parser.add_argument("--num_samples_val_test", type=int, default=75)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    generate_nicopp_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_prefix=args.metadata_prefix,
        download=args.download,
        seed=args.split_seed,
        num_samples_val_test=args.num_samples_val_test,
        verbose=True,
    )
