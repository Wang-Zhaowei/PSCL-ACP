
import argparse
DATA_TYPE = "multiclass"
DEFAULT_TRAIN_PATTERN = "datasets/ACP0819_10fold/train_{fold}.tsv"
DEFAULT_TEST_PATTERN = "datasets/ACP0819_10fold/test_{fold}.tsv"
DEFAULT_PDB_DIR = "datasets/ACP_0819/ACP0819_pdb_esmfold"
MODEL_SEQUENCE = ("SSCR", "SHTL", "CCI")


def run_multilabel(
    folds=range(1, 11),
    train_pattern=DEFAULT_TRAIN_PATTERN,
    test_pattern=DEFAULT_TEST_PATTERN,
    pdb_dir=DEFAULT_PDB_DIR,
    base_lr=2e-6,
    run_training=True,
    run_prediction=True,
):

    from torch.utils.data import random_split

    from data import MyDataset
    from runner import predict_model, train_model

    for fold in folds:
        train_dataset = None
        test_dataset = None


        if run_training:
            train_dataset = MyDataset(
                data_dir=train_pattern.format(fold=fold, idx=fold),
                pdb_dir=pdb_dir,
                data_type=DATA_TYPE,
            )

        if run_training or run_prediction:
            test_dataset = MyDataset(
                data_dir=test_pattern.format(fold=fold, idx=fold),
                pdb_dir=pdb_dir,
                data_type=DATA_TYPE,
            )

        if run_training:
            split_train_dataset, val_dataset = random_split(
                train_dataset,
                [0.8, 0.2],
            )
            for model_name in MODEL_SEQUENCE:
                print(f"Fold {fold}: training {model_name}")
                train_model(
                    split_train_dataset,
                    val_dataset,
                    DATA_TYPE,
                    idx=fold,
                    base_lr=base_lr,
                    model_name=model_name,
                )

        if run_prediction:
            for model_name in MODEL_SEQUENCE:
                print(f"Fold {fold}: predicting {model_name}")
                predict_model(
                    test_dataset,
                    DATA_TYPE,
                    model_name=model_name,
                    idx=fold,
                )


def build_parser():
    parser = argparse.ArgumentParser(
        description=" SSCR、SHTL、CCI "
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="10",
    )
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--test-pattern", default=DEFAULT_TEST_PATTERN)
    parser.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    parser.add_argument("--base-lr", type=float, default=2e-6)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--predict-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_multilabel(
        folds=args.folds,
        train_pattern=args.train_pattern,
        test_pattern=args.test_pattern,
        pdb_dir=args.pdb_dir,
        base_lr=args.base_lr,
        run_training=not args.predict_only,
        run_prediction=not args.train_only,
    )


if __name__ == "__main__":
    main()
