from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np, pandas as pd
from src.common.sandbox import copy_allowed_assets
from src.common.types import TaskContext, TaskDifficulty


@dataclass
class Task:
    task_id: str
    name: str
    difficulty: Union[str, TaskDifficulty]
    description: str
    dataset_path: Path
    evaluation_metric: str
    timeout_seconds: int = 60
    target_column: str = "target"
    success_threshold: float = 0.5
    metric_direction: str = "higher_is_better"
    metadata: Dict[str, Any] = field(default_factory=dict)
    grader_func: Optional[Callable[[np.ndarray, np.ndarray, Dict[str, Any]], float]] = None

    def __post_init__(self):
        if isinstance(self.difficulty, str):
            self.difficulty = TaskDifficulty.from_str(self.difficulty)
        self.dataset_path = Path(self.dataset_path)

    def setup_environment(self, workspace_dir: Path, run_id: str) -> TaskContext:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        copied = copy_allowed_assets({
            "train": self.dataset_path / "train.csv",
            "test_features": self.dataset_path / "test_features.csv",
        }, workspace_dir)

        (workspace_dir / "TASK_README.md").write_text(f"# {self.name}\n{self.description}")

        return TaskContext(
            task_id=self.task_id, name=self.name, difficulty=self.difficulty,
            description=self.description, evaluation_metric=self.evaluation_metric,
            timeout_seconds=self.timeout_seconds, workspace_dir=workspace_dir,
            train_data_path=copied["train"], test_features_path=copied["test_features"],
            metadata={**self.metadata, "target_column": self.target_column, "success_threshold": self.success_threshold},
        )

    def load_ground_truth(self) -> np.ndarray:
        df = pd.read_csv(self.dataset_path / "test_labels.csv")
        return df[self.target_column].to_numpy(dtype=float) if self.target_column in df else df.iloc[:, 0].to_numpy(dtype=float)

    def grade(self, y_pred: Union[np.ndarray, str, Path]) -> Tuple[float, bool, Dict[str, Any]]:
        y_true = self.load_ground_truth()
        if isinstance(y_pred, (str, Path)):
            df = pd.read_csv(Path(y_pred))
            col = "prediction" if "prediction" in df else ("target" if "target" in df else df.columns[0])
            pred_arr = df[col].to_numpy(dtype=float)
        else:
            pred_arr = np.asarray(y_pred, dtype=float).ravel()

        score = float(self.grader_func(y_true, pred_arr, self.metadata))
        passed = (score >= self.success_threshold) if self.metric_direction == "higher_is_better" else (score <= self.success_threshold)
        return score, passed, {"score": round(score, 6), "threshold": self.success_threshold, "passed": passed}