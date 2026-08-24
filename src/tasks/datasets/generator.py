"""Deterministic benchmark dataset generator for mle-eval tasks."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def generate_low_preprocessing_dataset(output_dir: Path, n_train=1000, n_test=300, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    x1, x2, x3 = rng.normal(10.0, 5.0, n_total), rng.exponential(2.0, n_total), rng.uniform(-10.0, 10.0, n_total)
    x4, x5 = rng.normal(100.0, 25.0, n_total), rng.gamma(2.0, 3.0, n_total)
    cat1 = rng.choice(["North", "South", "East", "West"], n_total, p=[0.4, 0.3, 0.2, 0.1])
    cat2 = rng.choice(["TypeA", "TypeB", "TypeC"], n_total)
    cat1_map = {"North": 2.5, "South": -1.5, "East": 0.5, "West": 4.0}
    cat2_map = {"TypeA": 1.0, "TypeB": -2.0, "TypeC": 3.0}
    cat1_eff = np.array([cat1_map[c] for c in cat1])
    cat2_eff = np.array([cat2_map[c] for c in cat2])
    y = 0.8*x1 - 1.2*x2 + 0.5*x3 + 0.05*x4 + 0.3*x5 + cat1_eff + cat2_eff + rng.normal(0, 0.2, n_total)
    df = pd.DataFrame({"feature_num1": x1, "feature_num2": x2, "feature_num3": x3, "feature_num4": x4, "feature_num5": x5, "category_region": cat1, "category_type": cat2, "target": y})
    df.loc[rng.random(n_total) < 0.12, "feature_num1"] = np.nan
    df.loc[rng.random(n_total) < 0.15, "feature_num3"] = np.nan
    df.iloc[:n_train].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)


def generate_low_tabular_regression_dataset(output_dir: Path, n_train=1000, n_test=300, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    X = rng.normal(0, 1, size=(n_total, 6))
    weights = np.array([2.5, -1.8, 0.9, -3.2, 1.4, -0.5])
    y = X @ weights + rng.normal(0, 0.2, size=n_total)
    train_outliers = rng.choice(n_train, size=int(n_train * 0.03), replace=False)
    y[train_outliers] += rng.uniform(8.0, 15.0, size=len(train_outliers))
    df = pd.DataFrame(X, columns=[f"x_{i+1}" for i in range(6)])
    df["target"] = y
    df.iloc[:n_train].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)


def generate_medium_imbalanced_fraud_dataset(output_dir: Path, n_train=2000, n_test=600, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n_total)
    device_risk = rng.beta(a=0.5, b=2.0, size=n_total)
    velocity_24h = rng.poisson(lam=3.0, size=n_total)
    ip_reputation = rng.uniform(0.0, 1.0, size=n_total)
    dist_from_home = rng.exponential(scale=15.0, size=n_total)
    v1, v2, v3 = rng.normal(0, 1, n_total), rng.normal(0, 1, n_total), rng.normal(0, 1, n_total)
    logits = -4.5 + 0.005*amount + 3.2*device_risk + 0.4*velocity_24h - 2.8*ip_reputation + 0.04*dist_from_home + 0.8*v1 - 0.6*v2 + 0.5*(device_risk*velocity_24h)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(size=n_total) < probs).astype(int)
    df = pd.DataFrame({"amount": amount, "device_risk": device_risk, "velocity_24h": velocity_24h, "ip_reputation": ip_reputation, "dist_from_home": dist_from_home, "v1": v1, "v2": v2, "v3": v3, "target": y})
    df.iloc[:n_train].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)


def generate_medium_hyperparam_dataset(output_dir: Path, n_train=1500, n_test=500, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    X = rng.uniform(0.0, 1.0, size=(n_total, 10))
    y = 10.0*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20.0*((X[:, 2]-0.5)**2) + 10.0*X[:, 3] + 5.0*X[:, 4] + rng.normal(0, 0.5, size=n_total)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["target"] = y
    df.iloc[:n_train].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)


def generate_high_multivariate_timeseries_dataset(output_dir: Path, n_train_steps=500, n_test_steps=48, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = n_train_steps + n_test_steps
    t = np.arange(total_steps)
    base = 50.0 + 0.02*t + 4.0*np.sin(2*np.pi*t/24) + 7.0*np.cos(2*np.pi*t/168)
    ch1 = base + rng.normal(0, 1.0, size=total_steps)
    ch2 = 0.6*np.roll(ch1, 1) + 2.0*np.sin(2*np.pi*t/12) + rng.normal(0, 0.8, size=total_steps)
    ch3 = 1.2*ch1 - 0.4*ch2 + rng.normal(0, 1.2, size=total_steps)
    ch4 = 0.5*np.roll(ch2, 2) + rng.normal(0, 0.5, size=total_steps)
    target = 0.5*ch1 + 0.3*ch2 + 0.2*ch3 + 0.1*ch4 + 3.0*np.sin(2*np.pi*t/24) + rng.normal(0, 0.5, size=total_steps)
    df = pd.DataFrame({"timestamp_idx": t, "sensor_ch1": ch1, "sensor_ch2": ch2, "sensor_ch3": ch3, "sensor_ch4": ch4, "target": target})
    df.iloc[:n_train_steps].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train_steps:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train_steps:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)


def generate_high_custom_loss_dataset(output_dir: Path, n_train=1200, n_test=400, seed=42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    price = rng.uniform(20.0, 100.0, size=n_total)
    discount = rng.beta(1.5, 5.0, size=n_total)
    comp_price = price + rng.normal(0, 5.0, size=n_total)
    season_idx = rng.uniform(0.5, 2.0, size=n_total)
    store_traffic = rng.poisson(lam=150, size=n_total)
    marketing_spend = rng.exponential(scale=500.0, size=n_total)
    noise = rng.normal(0, 1.0, size=n_total) * (5.0 + 0.1*price + 0.02*store_traffic)
    demand = np.maximum(120.0 - 1.2*price + 80.0*discount + 0.8*(comp_price - price) + 25.0*season_idx + 0.4*store_traffic + 0.03*marketing_spend + noise, 1.0)
    df = pd.DataFrame({"price": price, "discount": discount, "comp_price": comp_price, "season_idx": season_idx, "store_traffic": store_traffic, "marketing_spend": marketing_spend, "target": demand})
    df.iloc[:n_train].to_csv(output_dir / "train.csv", index=False)
    df.iloc[n_train:].drop(columns=["target"]).to_csv(output_dir / "test_features.csv", index=False)
    df.iloc[n_train:][["target"]].to_csv(output_dir / "test_labels.csv", index=False)