# Phishing URL Detection — AI in Cybersecurity Lab 2

A baseline machine learning project for detecting phishing URLs using extracted URL features. This project trains a Random Forest classifier on synthetic URL feature data and evaluates its performance using standard classification metrics.

## Dataset

The dataset consists of **2,000 synthetic URL samples** (1,000 benign, 1,000 phishing) with the following extracted features:

| Feature | Description |
|---|---|
| `url_length` | Total character length of the URL |
| `num_dots` | Number of dots in the URL |
| `has_https` | Whether the URL uses HTTPS (0/1) |
| `has_ip_address` | Whether the URL contains an IP address (0/1) |
| `num_special_chars` | Count of special characters (@, !, #, $, etc.) |
| `path_length` | Length of the URL path component |
| `num_subdomains` | Number of subdomains in the URL |
| `has_suspicious_words` | Contains words like 'login', 'verify', 'update' (0/1) |
| `domain_length` | Length of the domain name |
| `num_digits_in_domain` | Count of numeric digits in the domain |

The synthetic data is generated on first run if no raw CSV is present. You can replace it with a real phishing URL dataset (e.g., from PhiUSIIL or UCI) by placing a CSV with the same column schema in `data/raw/phishing_urls.csv`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python -m src.train
```

### Evaluate the model

```bash
python -m src.eval
```

### Run EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Project Structure

```
ai-cyber-lab2/
    README.md
    requirements.txt
    .gitignore
    data/
        raw/                  # Raw dataset (gitignored)
        processed/            # Cleaned features
    notebooks/
        01_eda.ipynb          # Exploratory data analysis
    src/
        __init__.py
        data.py               # Data loading and preprocessing
        train.py              # Model training
        eval.py               # Model evaluation
        utils.py              # Helper functions
    results/
        metrics.json          # Evaluation metrics
        confusion_matrix.png  # Confusion matrix plot
```

## Baseline Results

| Metric | Score |
|---|---|
| Accuracy | 0.9925 |
| Precision | 0.9900 |
| Recall | 0.9950 |
| F1-Score | 0.9925 |

*Exact results are saved in `results/metrics.json` after running the evaluation pipeline.*

## Ethics and Safety Considerations

- **Intended use**: This project is strictly for educational purposes as part of a cybersecurity course. It demonstrates how ML can be applied to detect phishing URLs.
- **Misuse risk**: Phishing detection models could theoretically be reverse-engineered to craft more evasive phishing URLs. This project uses synthetic data and a simple model, limiting any real-world misuse potential.
- **Bias and fairness**: The synthetic dataset is balanced, but real-world phishing data is heavily imbalanced. A production system would need careful handling of class imbalance and ongoing retraining as phishing techniques evolve.
- **Privacy**: No real user data or actual phishing URLs are used in this project. All data is synthetically generated.
- **Limitations**: This baseline model is not suitable for production deployment. Real phishing detection requires continually updated datasets, more sophisticated feature engineering, and ensemble approaches.
