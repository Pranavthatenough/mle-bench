import numpy as np, pandas as pd
from src.agents.base import BaseAgent
from src.agents.loader import register_agent
from src.common.types import SolutionArtifact, TaskContext

@register_agent("dummy")
class DummyMLAgent(BaseAgent):
    def __init__(self, name="DummyMLAgent", model_name="baseline-constant", version="0.1.0", **kwargs):
        super().__init__(name=name, model_name=model_name, version=version, **kwargs)

    def solve(self, task_context: TaskContext) -> SolutionArtifact:
        self.record_usage(tokens=120, cost_usd=0.0002)
        n = len(pd.read_csv(task_context.test_features_path))
        preds = np.zeros(n, dtype=float)
        p_file = task_context.workspace_dir / "predictions.csv"
        pd.DataFrame({"prediction": preds}).to_csv(p_file, index=False)
        return SolutionArtifact(predictions_file=p_file, predictions=preds.tolist())