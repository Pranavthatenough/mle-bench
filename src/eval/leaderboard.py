import json
from pathlib import Path
from typing import Any, Dict, List, Union
from src.common.types import BenchmarkRunReport

class LeaderboardAggregator:
    def __init__(self, results_dir: Union[str, Path] = "results"):
        self.results_dir = Path(results_dir)

    def load_all_reports(self) -> List[BenchmarkRunReport]:
        if not self.results_dir.exists(): return []
        reports = []
        for f in sorted(self.results_dir.glob("*.json")):
            try: reports.append(BenchmarkRunReport.from_dict(json.loads(f.read_text())))
            except Exception: pass
        return reports

    def aggregate(self) -> List[Dict[str, Any]]:
        reports = self.load_all_reports()
        reports.sort(key=lambda r: r.timestamp, reverse=True)
        seen, rows = set(), []
        for rep in reports:
            k = (rep.agent_name, rep.model_name)
            if k in seen: continue
            seen.add(k)
            s = rep.summary
            rows.append({
                "agent_name": rep.agent_name, "base_llm": rep.model_name,
                "low_diff": f"{s.get('low_diff_success_pct', 0.0):.1f}%",
                "med_diff": f"{s.get('med_diff_success_pct', 0.0):.1f}%",
                "high_diff": f"{s.get('high_diff_success_pct', 0.0):.1f}%",
                "overall_success": f"{s.get('overall_success_pct', 0.0):.1f}%",
                "overall_val": s.get("overall_success_pct", 0.0),
                "avg_runtime": f"{s.get('avg_runtime_seconds', 0.0):.2f}s",
                "cost_tokens": f"${s.get('total_cost_usd', 0.0):.4f} / {s.get('total_tokens', 0):,} tok",
            })
        rows.sort(key=lambda r: (-r["overall_val"], r["avg_runtime"]))
        return rows

    def render_terminal_table(self) -> str:
        rows = self.aggregate()
        if not rows: return "No results found."
        headers = ["Rank", "Agent Name", "Base LLM", "Low Diff (%)", "Med Diff (%)", "High Diff (%)", "Overall (%)", "Avg Time", "Cost / Tokens"]
        table = [[str(i), r["agent_name"], r["base_llm"], r["low_diff"], r["med_diff"], r["high_diff"], r["overall_success"], r["avg_runtime"], r["cost_tokens"]] for i, r in enumerate(rows, 1)]
        widths = [max(len(str(row[i])) for row in [headers] + table) for i in range(len(headers))]
        sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
        head = "| " + " | ".join([headers[i].ljust(widths[i]) for i in range(len(headers))]) + " |"
        body = ["| " + " | ".join([cell.ljust(widths[i]) for i, cell in enumerate(row)]) + " |" for row in table]
        return "\n".join([sep, head, sep] + body + [sep])

    def export_markdown(self, output_path: Union[str, Path] = "LEADERBOARD.md") -> Path:
        rows = self.aggregate()
        headers = ["Rank", "Agent Name", "Base LLM", "Low Diff (%)", "Medium Diff (%)", "High Diff (%)", "Overall Success (%)", "Avg Runtime (s)", "Total Cost / Tokens"]
        lines = ["# mle-eval Official Leaderboard\n", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for i, r in enumerate(rows, 1):
            lines.append(f"| {i} | **{r['agent_name']}** | `{r['base_llm']}` | {r['low_diff']} | {r['med_diff']} | {r['high_diff']} | **{r['overall_success']}** | {r['avg_runtime']} | {r['cost_tokens']} |")
        p = Path(output_path)
        p.write_text("\n".join(lines) + "\n")
        return p