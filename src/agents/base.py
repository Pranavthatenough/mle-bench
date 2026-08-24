from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.common.types import SolutionArtifact, TaskContext

class BaseAgent(ABC):
    def __init__(self, name: Optional[str] = None, model_name: str = "custom-llm", version: str = "1.0.0", **kwargs: Any):
        self._name = name if name is not None else self.__class__.__name__
        self._model_name = model_name
        self._version = version
        self.metadata = kwargs
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def name(self) -> str: return self._name
    @property
    def model_name(self) -> str: return self._model_name
    @property
    def version(self) -> str: return self._version

    def record_usage(self, tokens: int = 0, cost_usd: float = 0.0):
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd

    @abstractmethod
    def solve(self, task_context: TaskContext) -> SolutionArtifact:
        raise NotImplementedError