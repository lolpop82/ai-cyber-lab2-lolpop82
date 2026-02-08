"""Train a baseline phishing detection model."""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data import load_data, get_splits
from src.utils import get_project_root, ensure_dir


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename="model.joblib"):
    """Save the trained model to the results directory."""
    root = get_project_root()
    results_dir = os.path.join(root, "results")
    ensure_dir(results_dir)
    filepath = os.path.join(results_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def main():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    print(f"Training on {len(X_train)} samples...")
    model = train_model(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    save_model(model)
    print("Training complete.")


if __name__ == "__main__":
    main()
