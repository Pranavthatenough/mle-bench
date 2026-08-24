from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TaskDifficulty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_str(cls, val: str) -> TaskDifficulty:
        return cls(val.lower().strip())


@dataclass
class TaskContext:
    task_id: str
    name: str
    difficulty: TaskDifficulty
    description: str
    evaluation_metric: str
    timeout_seconds: int
    workspace_dir: Path
    train_data_path: Path
    test_features_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_submission_path: Optional[Path] = None


@dataclass
class SolutionArtifact:
    code: Optional[str] = None
    predictions_file: Optional[Path] = None
    predictions: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_stdout: Optional[str] = None
    execution_stderr: Optional[str] = None


@dataclass
class Telemetry:
    execution_time_seconds: float = 0.0
    memory_overhead_mb: float = 0.0
    exit_code: Optional[int] = 0
    timeout_occurred: bool = False
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    task_id: str
    task_name: str
    difficulty: str
    metric_name: str
    score: float
    threshold: float
    passed: bool
    telemetry: Telemetry
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "metric_name": self.metric_name,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "passed": self.passed,
            "telemetry": self.telemetry.to_dict(),
            "details": self.details,
        }


@dataclass
class BenchmarkRunReport:
    agent_name: str
    model_name: str
    agent_version: str
    timestamp: str
    run_id: str
    results: List[EvalResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "agent_version": self.agent_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkRunReport:
        results = []
        for r in data.get("results", []):
            telem_data = r.get("telemetry", {})
            telemetry = Telemetry(
                execution_time_seconds=telem_data.get("execution_time_seconds", 0.0),
                memory_overhead_mb=telem_data.get("memory_overhead_mb", 0.0),
                exit_code=telem_data.get("exit_code"),
                timeout_occurred=telem_data.get("timeout_occurred", False),
                error_message=telem_data.get("error_message"),
                tokens_used=telem_data.get("tokens_used"),
                cost_usd=telem_data.get("cost_usd"),
            )
            results.append(
                EvalResult(
                    task_id=r.get("task_id", ""),
                    task_name=r.get("task_name", ""),
                    difficulty=r.get("difficulty", "low"),
                    metric_name=r.get("metric_name", ""),
                    score=r.get("score", 0.0),
                    threshold=r.get("threshold", 0.0),
                    passed=r.get("passed", False),
                    telemetry=telemetry,
                    details=r.get("details", {}),
                )
            )
        return cls(
            agent_name=data.get("agent_name", "UnknownAgent"),
            model_name=data.get("model_name", "UnknownModel"),
            agent_version=data.get("agent_version", "1.0.0"),
            timestamp=data.get("timestamp", ""),
            run_id=data.get("run_id", ""),
            results=results,
            summary=data.get("summary", {}),
        )