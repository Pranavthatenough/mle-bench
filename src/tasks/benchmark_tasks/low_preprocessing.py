from pathlib import Path
from src.common.metrics import r2_score
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_low_preprocessing_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_low_01"
if not (DATA_DIR / "train.csv").exists():
    generate_low_preprocessing_dataset(DATA_DIR)

TASK_LOW_PREPROCESSING = Task(
    task_id="task_low_01", name="Feature Preprocessing & Imputation", difficulty=TaskDifficulty.LOW,
    description="Impute missing numbers and encode categories.", dataset_path=DATA_DIR,
    evaluation_metric="R2_Score", timeout_seconds=30, success_threshold=0.80, metric_direction="higher_is_better",
    grader_func=lambda yt, yp, meta: r2_score(yt, yp)
)
TaskRegistry.register(TASK_LOW_PREPROCESSING)