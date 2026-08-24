from pathlib import Path
from src.common.metrics import root_mean_squared_error
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_medium_hyperparam_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_med_02"
if not (DATA_DIR / "train.csv").exists():
    generate_medium_hyperparam_dataset(DATA_DIR)

TASK_MEDIUM_HYPERPARAM = Task(
    task_id="task_med_02", name="Nonlinear Regression & Tuning", difficulty=TaskDifficulty.MEDIUM,
    description="Friedman non-linear regression challenge.", dataset_path=DATA_DIR,
    evaluation_metric="RMSE", timeout_seconds=45, success_threshold=1.20, metric_direction="lower_is_better",
    grader_func=lambda yt, yp, meta: root_mean_squared_error(yt, yp)
)
TaskRegistry.register(TASK_MEDIUM_HYPERPARAM)