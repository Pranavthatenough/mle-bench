import concurrent.futures, datetime, time, uuid
from pathlib import Path
from typing import List, Optional, Union
from src.agents.base import BaseAgent
from src.common.sandbox import create_isolated_workspace
from src.common.types import BenchmarkRunReport, EvalResult, SolutionArtifact, Telemetry
from src.eval.grader import Grader
from src.eval.runner import SubprocessRunner
from src.tasks.base import Task

class EvaluationHarness:
    def __init__(self, results_dir: Optional[Union[str, Path]] = None, workspace_base_dir: Optional[Union[str, Path]] = None):
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.workspace_base_dir = workspace_base_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_single_task(self, agent: BaseAgent, task: Task, run_id: str) -> EvalResult:
        ws = create_isolated_workspace(task.task_id, run_id, self.workspace_base_dir)
        ctx = task.setup_environment(ws, run_id)
        start = time.perf_counter()
        telemetry = Telemetry()
        try:
            sol: SolutionArtifact = agent.solve(ctx)
            script_file = ws / "solution.py"
            preds_file = ws / "predictions.csv"
            if script_file.exists() and not preds_file.exists():
                telemetry, _, _ = SubprocessRunner.execute_script(script_file, ws, task.timeout_seconds)
            else:
                telemetry.execution_time_seconds = round(time.perf_counter() - start, 3)
            score, passed, details = Grader.evaluate_solution(task, preds_file)
        except Exception as e:
            telemetry.execution_time_seconds = round(time.perf_counter() - start, 3)
            telemetry.error_message = str(e)
            score, passed, details = 0.0, False, {"error": str(e)}

        telemetry.tokens_used = agent.total_tokens
        telemetry.cost_usd = agent.total_cost_usd
        return EvalResult(task_id=task.task_id, task_name=task.name, difficulty=task.difficulty.value, metric_name=task.evaluation_metric, score=score, threshold=task.success_threshold, passed=passed, telemetry=telemetry, details=details)

    def run_benchmark(self, agent: BaseAgent, tasks: List[Task], parallel: int = 1) -> BenchmarkRunReport:
        run_id = f"run_{agent.name.lower()}_{uuid.uuid4().hex[:6]}"
        results: List[EvalResult] = []
        if parallel > 1 and len(tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(self.evaluate_single_task, agent, t, run_id) for t in tasks]
                results = [f.result() for f in futures]
        else:
            results = [self.evaluate_single_task(agent, t, run_id) for t in tasks]
        results.sort(key=lambda r: r.task_id)

        low = [r for r in results if r.difficulty == "low"]
        med = [r for r in results if r.difficulty == "medium"]
        high = [r for r in results if r.difficulty == "high"]
        summary = {
            "total_tasks": len(results),
            "passed_tasks": sum(1 for r in results if r.passed),
            "low_diff_success_pct": round((sum(1 for r in low if r.passed)/len(low)*100) if low else 0, 1),
            "med_diff_success_pct": round((sum(1 for r in med if r.passed)/len(med)*100) if med else 0, 1),
            "high_diff_success_pct": round((sum(1 for r in high if r.passed)/len(high)*100) if high else 0, 1),
            "overall_success_pct": round((sum(1 for r in results if r.passed)/len(results)*100) if results else 0, 1),
            "avg_runtime_seconds": round(sum(r.telemetry.execution_time_seconds for r in results)/len(results), 3) if results else 0,
            "total_tokens": agent.total_tokens,
            "total_cost_usd": round(agent.total_cost_usd, 4),
        }
        report = BenchmarkRunReport(agent_name=agent.name, model_name=agent.model_name, agent_version=agent.version, timestamp=datetime.datetime.now().isoformat(), run_id=run_id, results=results, summary=summary)
        report.save_json(self.results_dir / f"{agent.name.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return report