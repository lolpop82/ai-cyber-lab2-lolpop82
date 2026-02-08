"""Compare multiple classifiers on the phishing detection task."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data import load_data, get_splits
from src.utils import get_project_root, ensure_dir, save_json


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
}


def evaluate_model(model, X_test, y_test):
    """Return metrics dict for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }


def plot_comparison(results, output_path):
    """Save a bar chart comparing model metrics."""
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = range(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar([xi + i * width for xi in x], values, width, label=metric)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Phishing URL Detection")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.8, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    results = {}
    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"  {metrics}")

    # Save comparison results
    root = get_project_root()
    results_dir = os.path.join(root, "results")
    ensure_dir(results_dir)

    save_json(results, os.path.join(results_dir, "model_comparison.json"))
    print("Saved results/model_comparison.json")

    plot_comparison(results, os.path.join(results_dir, "model_comparison.png"))
    print("Saved results/model_comparison.png")

    # Print winner
    best = max(results, key=lambda m: results[m]["f1_score"])
    print(f"\nBest model by F1: {best} ({results[best]['f1_score']})")


if __name__ == "__main__":
    main()
