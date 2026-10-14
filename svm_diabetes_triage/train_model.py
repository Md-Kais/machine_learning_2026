"""Train the binary linear classifier; algorithm code lives in svm_model.py."""
import argparse
from pathlib import Path
from svm_model import SimpleLinearSVM, load_dataset

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train a linear SVM using SMO.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data/diabetes_triage.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "dist/model.json")
    parser.add_argument("--C", type=float, default=10.0, help="Penalty for margin violations.")
    parser.add_argument("--tolerance", type=float, default=0.0001)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    model = SimpleLinearSVM(args.C, args.tolerance, args.max_passes, args.max_iterations, args.random_seed).fit(load_dataset(args.dataset))
    model.export_json(args.output)
    print(f"Loaded {len(model.training_examples)} examples. Training accuracy: {model.training_accuracy():.1%}")
    print(f"w = {model.weights}, b = {model.bias:.6f}; support vectors: {len(model.support_indices())}")
    print(f"Converged: {model.converged}; iterations: {model.iterations}; KKT violation: {model.kkt_violation:.6g}")
    if not model.converged:
        print("Increase --max-iterations or review --tolerance before relying on this approximate fit.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
