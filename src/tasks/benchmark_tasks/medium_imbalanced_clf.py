from pathlib import Path
import numpy as np
from src.common.metrics import f1_score
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_medium_imbalanced_fraud_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_med_01"
if not (DATA_DIR / "train.csv").exists():
    generate_medium_imbalanced_fraud_dataset(DATA_DIR)

TASK_MEDIUM_IMBALANCED = Task(
    task_id="task_med_01",
    name="Imbalanced Fraud Classification",
    difficulty=TaskDifficulty.MEDIUM,
    description="Imbalanced fraud detection evaluated with F1-Score.",
    dataset_path=DATA_DIR,
    evaluation_metric="F1_Score",
    timeout_seconds=45,
    target_column="target",
    success_threshold=0.15,
    metric_direction="higher_is_better",
    grader_func=lambda yt, yp, meta: f1_score(yt, (yp >= 0.5).astype(int)),
)
TaskRegistry.register(TASK_MEDIUM_IMBALANCED)