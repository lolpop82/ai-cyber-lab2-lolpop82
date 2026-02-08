"""Data loading, preprocessing, and train/test splitting for phishing detection."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_project_root, ensure_dir


def generate_synthetic_dataset(n_samples=2000, random_state=42):
    """Generate a synthetic phishing URL feature dataset.

    Features are based on common URL characteristics used in phishing detection:
    - url_length: total length of the URL
    - num_dots: number of dots in the URL
    - has_https: whether the URL uses HTTPS (1 or 0)
    - has_ip_address: whether the URL contains an IP address (1 or 0)
    - num_special_chars: count of special characters (@, !, #, $, etc.)
    - path_length: length of the URL path component
    - num_subdomains: number of subdomains
    - has_suspicious_words: whether URL contains words like 'login', 'verify', 'update' (1 or 0)
    - domain_length: length of the domain name
    - num_digits_in_domain: count of digits in the domain
    """
    rng = np.random.RandomState(random_state)

    n_benign = n_samples // 2
    n_phishing = n_samples - n_benign

    # Benign URLs tend to have shorter lengths, fewer special chars, HTTPS, etc.
    benign = pd.DataFrame({
        "url_length": rng.normal(45, 15, n_benign).clip(10, 200).astype(int),
        "num_dots": rng.poisson(2, n_benign).clip(1, 10),
        "has_https": rng.binomial(1, 0.85, n_benign),
        "has_ip_address": rng.binomial(1, 0.02, n_benign),
        "num_special_chars": rng.poisson(1, n_benign).clip(0, 15),
        "path_length": rng.normal(15, 8, n_benign).clip(0, 100).astype(int),
        "num_subdomains": rng.poisson(1, n_benign).clip(0, 5),
        "has_suspicious_words": rng.binomial(1, 0.05, n_benign),
        "domain_length": rng.normal(10, 4, n_benign).clip(3, 40).astype(int),
        "num_digits_in_domain": rng.poisson(0.5, n_benign).clip(0, 10),
        "label": 0,
    })

    # Phishing URLs tend to be longer, more special chars, use IP addresses, etc.
    phishing = pd.DataFrame({
        "url_length": rng.normal(80, 25, n_phishing).clip(20, 300).astype(int),
        "num_dots": rng.poisson(4, n_phishing).clip(1, 15),
        "has_https": rng.binomial(1, 0.35, n_phishing),
        "has_ip_address": rng.binomial(1, 0.25, n_phishing),
        "num_special_chars": rng.poisson(4, n_phishing).clip(0, 20),
        "path_length": rng.normal(35, 15, n_phishing).clip(0, 150).astype(int),
        "num_subdomains": rng.poisson(3, n_phishing).clip(0, 8),
        "has_suspicious_words": rng.binomial(1, 0.45, n_phishing),
        "domain_length": rng.normal(20, 8, n_phishing).clip(5, 60).astype(int),
        "num_digits_in_domain": rng.poisson(3, n_phishing).clip(0, 15),
        "label": 1,
    })

    df = pd.concat([benign, phishing], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def load_data():
    """Load the phishing dataset. Generates synthetic data if no raw file exists."""
    root = get_project_root()
    raw_path = os.path.join(root, "data", "raw", "phishing_urls.csv")
    processed_path = os.path.join(root, "data", "processed", "phishing_features.csv")

    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        print("No raw data found. Generating synthetic phishing URL dataset...")
        df = generate_synthetic_dataset()
        ensure_dir(os.path.dirname(raw_path))
        df.to_csv(raw_path, index=False)
        print(f"Saved synthetic dataset to {raw_path}")

    # Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # Save processed data
    ensure_dir(os.path.dirname(processed_path))
    df.to_csv(processed_path, index=False)

    return df


def get_splits(df, test_size=0.2, random_state=42):
    """Split data into train/test sets and return feature matrices and labels."""
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    X_train, X_test, y_train, y_test = get_splits(df)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
