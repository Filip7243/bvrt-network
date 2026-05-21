import argparse
from pathlib import Path

from train import run_loso_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train Siamese ResNet18 geometry-fusion BVRT model.")
    parser.add_argument("--root-dir", default="data/processed/siemens-net-data")
    parser.add_argument("--results-dir", default="results/siamese-resnet18-vector-geometry")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-dropout", type=float, default=0.0)
    parser.add_argument("--include-raw-features", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fine-tune-backbone", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    if not Path(root_dir).exists() and Path("../data/processed/siemens-net-data").exists():
        root_dir = "../data/processed/siemens-net-data"

    run_loso_training(
        root_dir=root_dir,
        num_epochs=args.num_epochs,
        results_dir=args.results_dir,
        spatial_dropout=args.spatial_dropout,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        seed=args.seed,
        include_raw_features=args.include_raw_features,
        pretrained=not args.no_pretrained,
        fine_tune_backbone=args.fine_tune_backbone,
        model_arch="resnet18_vector_geometry_fusion",
        use_semantic_augmentation=not args.no_augmentation,
        max_folds=args.max_folds,
    )
