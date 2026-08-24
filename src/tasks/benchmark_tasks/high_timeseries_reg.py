from pathlib import Path
import pandas as pd
from src.common.metrics import mean_absolute_scaled_error
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_high_multivariate_timeseries_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_high_01"
if not (DATA_DIR / "train.csv").exists():
    generate_high_multivariate_timeseries_dataset(DATA_DIR)

def grade_ts(yt, yp, meta):
    y_tr = pd.read_csv(DATA_DIR / "train.csv")["target"].to_numpy(dtype=float)
    return mean_absolute_scaled_error(yt, yp, y_tr, period=24)

TASK_HIGH_TIMESERIES = Task(
    task_id="task_high_01", name="Multi-Variate Time-Series Forecasting", difficulty=TaskDifficulty.HIGH,
    description="Sensor time-series with seasonality.", dataset_path=DATA_DIR,
    evaluation_metric="MASE", timeout_seconds=60, success_threshold=0.85, metric_direction="lower_is_better",
    grader_func=grade_ts
)
TaskRegistry.register(TASK_HIGH_TIMESERIES)