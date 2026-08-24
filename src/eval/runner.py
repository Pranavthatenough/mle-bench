import sys
from pathlib import Path
from src.common.sandbox import run_solution_subprocess

class SubprocessRunner:
    @classmethod
    def execute_script(cls, script_path: Path, workspace_dir: Path, timeout_seconds: int = 60):
        cmd = [sys.executable, str(script_path.name)]
        return run_solution_subprocess(command=cmd, cwd=workspace_dir, timeout_seconds=timeout_seconds)