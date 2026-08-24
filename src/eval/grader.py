from pathlib import Path
from typing import Any, Tuple, Union
import numpy as np, pandas as pd
from src.tasks.base import Task

class Grader:
    @classmethod
    def evaluate_solution(cls, task: Task, predictions_or_file: Union[np.ndarray, list, Path, str]) -> Tuple[float, bool, Any]:
        return task.grade(predictions_or_file)