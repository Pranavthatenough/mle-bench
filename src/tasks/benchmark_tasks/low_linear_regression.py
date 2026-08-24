from pathlib import Path
from src.common.metrics import root_mean_squared_error
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_low_tabular_regression_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_low_02"
if not (DATA_DIR / "train.csv").exists():
    generate_low_tabular_regression_dataset(DATA_DIR, n_train=1000, n_test=300, seed=42)

TASK_LOW_REGRESSION = Task(
    task_id="task_low_02",
    name="Robust Tabular Regression",
    difficulty=TaskDifficulty.LOW,
    description="Continuous regression dataset with heavy-tailed outlier noise in training.",
    dataset_path=DATA_DIR,
    evaluation_metric="RMSE",
    timeout_seconds=30,
    target_column="target",
    success_threshold=0.60,
    metric_direction="lower_is_better",
    grader_func=lambda yt, yp, meta: root_mean_squared_error(yt, yp),
)
TaskRegistry.register(TASK_LOW_REGRESSION)