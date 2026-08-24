import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.loader import AGENT_REGISTRY, discover_agents_in_dir, load_agent
from src.eval.harness import EvaluationHarness
from src.eval.leaderboard import LeaderboardAggregator
from src.tasks.registry import TaskRegistry


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mle-eval", description="mle-eval Benchmark Framework")
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Run benchmark on an agent")
    run_p.add_argument("--agent", "-a", type=str, required=True, help="Agent name (e.g. 'heuristic', 'dummy')")
    run_p.add_argument("--difficulty", "-d", type=str, default="all", choices=["all", "low", "medium", "high"])
    run_p.add_argument("--tasks", "-t", type=str, default=None, help="Comma-separated task IDs")
    run_p.add_argument("--parallel", "-p", type=int, default=1, help="Number of parallel workers")
    run_p.add_argument("--results-dir", "-r", type=str, default="results")

    lb_p = subparsers.add_parser("leaderboard", help="Show leaderboard")
    lb_p.add_argument("--results-dir", "-r", type=str, default="results")
    lb_p.add_argument("--export", "-e", type=str, default="table", choices=["table", "markdown"])
    lb_p.add_argument("--output", "-o", type=str, default="LEADERBOARD.md")

    subparsers.add_parser("list-tasks", help="List all benchmark tasks")
    subparsers.add_parser("list-agents", help="List all available agents")

    return parser


def main():
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "run":
        try:
            agent = load_agent(args.agent)
        except Exception as e:
            print(f"[Error] Could not load agent '{args.agent}': {e}")
            sys.exit(1)

        tasks = TaskRegistry.list_tasks(difficulty=args.difficulty)
        if args.tasks:
            sel = [t.strip() for t in args.tasks.split(",")]
            tasks = [t for t in tasks if t.task_id in sel]

        print(f"\n============================================================")
        print(f"  Running Benchmark for Agent: {agent.name}")
        print(f"============================================================")
        
        harness = EvaluationHarness(results_dir=args.results_dir)
        report = harness.run_benchmark(agent=agent, tasks=tasks, parallel=args.parallel)

        print("\nTask Results:")
        print("------------------------------------------------------------")
        for r in report.results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"{status} {r.task_id} ({r.difficulty}): {r.metric_name} = {r.score:.4f} (Threshold: {r.threshold})")
            if not r.passed and r.telemetry.error_message:
                print(f"       Reason: {r.telemetry.error_message}")

        print("------------------------------------------------------------")
        s = report.summary
        print(f"Summary: {s['passed_tasks']}/{s['total_tasks']} Passed ({s['overall_success_pct']}%) | Avg Time: {s['avg_runtime_seconds']}s")
        print("============================================================\n")

    elif args.command == "leaderboard":
        agg = LeaderboardAggregator(results_dir=args.results_dir)
        if args.export == "markdown":
            agg.export_markdown(args.output)
            print(f"Exported to {args.output}")
        else:
            print("\n" + agg.render_terminal_table() + "\n")
    elif args.command == "list-tasks":
        for t in TaskRegistry.list_tasks():
            print(f"[{t.difficulty.value.upper()}] {t.task_id}: {t.name} (Metric: {t.evaluation_metric})")
    elif args.command == "list-agents":
        discover_agents_in_dir(Path(__file__).parent / "agents")
        for k in sorted(AGENT_REGISTRY.keys()):
            print(f" • {k}")


if __name__ == "__main__":
    main()