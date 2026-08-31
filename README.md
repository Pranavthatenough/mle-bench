# mle-bench

**Autonomous AI ML Engineering Benchmark Framework**

`mle-bench` is an extensible, modular Python framework for evaluating and ranking autonomous AI agents on machine learning engineering tasks. It provides sandboxed execution, deterministic grading, and cost/telemetry tracking so agent performance can be compared on a level playing field.

---

## Table of Contents

- [Key Features](#key-features)
- [Benchmark Challenges](#benchmark-challenges)
- [Quickstart](#quickstart)
  - [Installation](#1-installation)
  - [Running a Benchmark](#2-running-a-benchmark)
  - [Viewing Results](#3-viewing-results)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Zero ground-truth data leakage** — sandboxed workspaces isolate training data and test features from true test labels, so agents cannot access held-out answers during a run.
- **Deterministic graders** — solutions are scored with standard metrics ($R^2$, RMSE, F1-Score, MASE, Asymmetric Loss), so repeated runs on the same submission produce the same score.
- **Cross-platform sandboxed execution** — a subprocess runner enforces per-task timeouts and tracks peak memory overhead, isolating agent code from the host environment.
- **Telemetry and cost tracking** — execution time, memory usage, token consumption, and estimated dollar cost are recorded for every run.
- **Live leaderboard** — results are shown as a terminal table during a run and exported to `LEADERBOARD.md` for tracking over time.

---

## Benchmark Challenges

| Task ID | Name | Tier | Target Metric | Pass Threshold | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `task_low_01` | Feature Preprocessing & Imputation | Low | $R^2$ Score | $\ge 0.80$ | Missing value imputation (MCAR/MAR), categorical encoding, feature scaling. |
| `task_low_02` | Robust Tabular Regression | Low | RMSE | $\le 0.60$ | Linear regression with heavy-tailed outlier noise in the training set. |
| `task_med_01` | Imbalanced Fraud Classification | Medium | F1-Score | $\ge 0.15$ | Financial transaction dataset with class imbalance (~5% fraud rate). |
| `task_med_02` | Nonlinear Hyperparameter Tuning | Medium | RMSE | $\le 1.20$ | Friedman #1 non-linear benchmark with sine and quadratic interactions. |
| `task_high_01` | Multi-Variate Time-Series Forecast | High | MASE | $\le 0.85$ | Multi-channel sensor forecasting with daily and weekly seasonality. |
| `task_high_02` | Asymmetric Custom Loss Demand | High | Asymmetric Loss | $\le 380.0$ | Demand estimation where under-predicting stock costs 4x more than over-stocking. |

Tiers are a rough indicator of difficulty: **Low**-tier tasks are single-step tabular problems solvable with standard preprocessing, **Medium**-tier tasks require handling imbalance or non-linear structure, and **High**-tier tasks require multi-step reasoning (time-series decomposition, custom loss handling) to pass.

---

## Quickstart

### 1. Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<your-org>/mle-bench.git
cd mle-bench
pip install -e .
```

### 2. Running a Benchmark

Run a single task against an agent:

```bash
mle-bench run --task task_med_01 --agent configs/agent.yaml
```

Run the full suite:

```bash
mle-bench run --all --agent configs/agent.yaml
```

Common flags:

| Flag | Description |
| :--- | :--- |
| `--task` | Task ID to run (see table above). Omit with `--all` to run every task. |
| `--agent` | Path to the agent config file (model, API key env var, sandbox limits). |
| `--timeout` | Override the default per-task timeout, in seconds. |
| `--seed` | Random seed for reproducibility. |

### 3. Viewing Results

Results print to the terminal as each task completes, and a full run summary is written to `LEADERBOARD.md`:

```bash
mle-bench leaderboard --export markdown
```

---

## Configuration

Agent behavior, model selection, and sandbox resource limits are defined in a YAML config file, e.g.:

```yaml
agent:
  name: my-agent
  model: gpt-4.1
  api_key_env: OPENAI_API_KEY

sandbox:
  timeout_seconds: 600
  max_memory_mb: 4096
```

See `configs/agent.example.yaml` for a full reference.

---

## Project Structure

```
mle-bench/
├── tasks/          # Task definitions: datasets, graders, pass thresholds
├── sandbox/        # Isolated subprocess execution environment
├── graders/        # Deterministic scoring implementations per metric
├── telemetry/      # Time, memory, token, and cost tracking
├── leaderboard/     # Terminal and Markdown leaderboard generation
├── configs/         # Example agent and task configs
└── README.md
```

---

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue first to discuss what you'd like to change, including any new task definitions or graders you're proposing.

---

## License

MIT