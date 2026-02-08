"""Evaluate the trained phishing detection model."""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.data import load_data, get_splits
from src.utils import get_project_root, ensure_dir, save_json


def load_model(filename="model.joblib"):
    """Load the trained model from the results directory."""
    root = get_project_root()
    filepath = os.path.join(root, "results", filename)
    return joblib.load(filepath)


def evaluate(model, X_test, y_test):
    """Compute evaluation metrics on the test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, output_path):
    """Save a confusion matrix plot as a PNG file."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign", "Phishing"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Phishing Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    print("Loading data and model...")
    df = load_data()
    _, X_test, _, y_test = get_splits(df)
    model = load_model()

    print(f"Evaluating on {len(X_test)} test samples...")
    metrics, y_pred = evaluate(model, X_test, y_test)

    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save metrics
    root = get_project_root()
    results_dir = os.path.join(root, "results")
    ensure_dir(results_dir)

    metrics_path = os.path.join(results_dir, "metrics.json")
    save_json(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # Save confusion matrix
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    save_confusion_matrix(y_test, y_pred, cm_path)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
