# mle-bench: Autonomous AI ML Engineering Benchmark Framework

`mle-bench` is an extensible, modular Python benchmarking framework designed to evaluate and rank autonomous AI agents on machine learning engineering challenges.

---

## 🚀 Key Features

* **Zero Ground-Truth Data Leakage**: Sandboxed workspaces isolate training data and test features from true test labels.
* **Deterministic Graders**: Evaluates solutions using standard metrics ($R^2$, RMSE, F1-Score, MASE, Asymmetric Loss).
* **Cross-Platform Sandboxed Execution**: Subprocess runner enforcing timeouts and tracking peak memory overhead.
* **Telemetry & Cost Tracking**: Measures execution time, memory usage, token consumption, and dollar costs.
* **Live Leaderboard**: Generates terminal tables and exports Markdown rankings to `LEADERBOARD.md`.

---

## 📊 Benchmark Challenges

| Task ID | Name | Tier | Target Metric | Pass Threshold | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `task_low_01` | Feature Preprocessing & Imputation | **LOW** | $R^2$ Score | $\ge 0.80$ | Missing value imputation (MCAR/MAR), categorical encoding, feature scaling. |
| `task_low_02` | Robust Tabular Regression | **LOW** | RMSE | $\le 0.60$ | Linear regression with heavy-tailed outlier noise in training set. |
| `task_med_01` | Imbalanced Fraud Classification | **MEDIUM** | F1-Score | $\ge 0.15$ | Financial transaction dataset with class imbalance (~5% fraud rate). |
| `task_med_02` | Nonlinear Hyperparameter Tuning | **MEDIUM** | RMSE | $\le 1.20$ | Friedman 1 non-linear benchmark with sine and quadratic interactions. |
| `task_high_01` | Multi-Variate Time-Series Forecast | **HIGH** | MASE | $\le 0.85$ | Multi-channel sensor forecasting with daily and weekly seasonality. |
| `task_high_02` | Asymmetric Custom Loss Demand | **HIGH** | Asymmetric Loss | $\le 380.0$ | Demand estimation where under-predicting stock costs 4x more than over-stocking. |

---

## ⚡ Quickstart

### 1. Installation
```bash
pip install -e .