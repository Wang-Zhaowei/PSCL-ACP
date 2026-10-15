"""
示例：
    python binary.py SSCR
    python binary.py SHTL
    python binary.py CCI

"""

import argparse


DATA_TYPE = "binary"
DEFAULT_TRAIN_DATA = "datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv"
DEFAULT_TEST_DATA = "datasets/ACP-Mixed-80/ACP-Mixed-80-test.tsv"
DEFAULT_TRAIN_PDB = "datasets/ACP-Mixed-80/train_pdb_esmfold"
DEFAULT_TEST_PDB = "datasets/ACP-Mixed-80/test_pdb_esmfold"


def run_binary(
    model_name,
    train_data=DEFAULT_TRAIN_DATA,
    test_data=DEFAULT_TEST_DATA,
    train_pdb=DEFAULT_TRAIN_PDB,
    test_pdb=DEFAULT_TEST_PDB,
    round_index=0,
    base_lr=2e-6,
    run_training=True,
    run_prediction=True,
):

    from torch.utils.data import random_split

    from data import MyDataset
    from runner import predict_model, train_model

    train_dataset = None
    test_dataset = None

    if run_training:
        train_dataset = MyDataset(
            data_dir=train_data,
            pdb_dir=train_pdb,
            data_type=DATA_TYPE,
        )

    if run_training or run_prediction:
        test_dataset = MyDataset(
            data_dir=test_data,
            pdb_dir=test_pdb,
            data_type=DATA_TYPE,
        )

    if run_training:
        split_train_dataset, val_dataset = random_split(
            train_dataset,
            [0.8, 0.2],
        )
        train_model(
            split_train_dataset,
            val_dataset,
            DATA_TYPE,
            idx=round_index,
            base_lr=base_lr,
            model_name=model_name,
        )

    if run_prediction:
        predict_model(
            test_dataset,
            DATA_TYPE,
            model_name=model_name,
            idx=round_index,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="binary")
    parser.add_argument(
        "model",
        type=str.upper,
        choices=("SSCR", "SHTL", "CCI"),
    )
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--train-pdb", default=DEFAULT_TRAIN_PDB)
    parser.add_argument("--test-pdb", default=DEFAULT_TEST_PDB)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=2e-6)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--predict-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_binary(
        model_name=args.model,
        train_data=args.train_data,
        test_data=args.test_data,
        train_pdb=args.train_pdb,
        test_pdb=args.test_pdb,
        round_index=args.round_index,
        base_lr=args.base_lr,
        run_training=not args.predict_only,
        run_prediction=not args.train_only,
    )


if __name__ == "__main__":
    main()
