from pathlib import Path
from src.common.metrics import asymmetric_weighted_loss
from src.common.types import TaskDifficulty
from src.tasks.base import Task
from src.tasks.datasets.generator import generate_high_custom_loss_dataset
from src.tasks.registry import TaskRegistry

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "task_high_02"
if not (DATA_DIR / "train.csv").exists():
    generate_high_custom_loss_dataset(DATA_DIR)

TASK_HIGH_CUSTOM_LOSS = Task(
    task_id="task_high_02", name="Asymmetric Custom Loss Demand Optimization", difficulty=TaskDifficulty.HIGH,
    description="Stock demand where under-predicting costs 4x more.", dataset_path=DATA_DIR,
    evaluation_metric="Asymmetric_Weighted_Loss", timeout_seconds=60, success_threshold=380.0, metric_direction="lower_is_better",
    grader_func=lambda yt, yp, meta: asymmetric_weighted_loss(yt, yp, under_prediction_weight=4.0, over_prediction_weight=1.0)
)
TaskRegistry.register(TASK_HIGH_CUSTOM_LOSS)