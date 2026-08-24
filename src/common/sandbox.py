"""Cross-platform workspace sandboxing and subprocess runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.common.types import Telemetry

# Safely import Unix resource module if available (avoids error on Windows)
try:
    import resource
except ImportError:
    resource = None


def create_isolated_workspace(
    task_id: str,
    run_id: str,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Creates a fresh, isolated workspace directory for a specific task evaluation run."""
    if base_dir is None:
        base_dir = Path(tempfile.gettempdir()) / "mle_eval_workspaces"
    else:
        base_dir = Path(base_dir)

    workspace = base_dir / run_id / task_id
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def copy_allowed_assets(
    source_assets: Dict[str, Path],
    destination_dir: Path,
) -> Dict[str, Path]:
    """Copies public assets into the workspace without ground-truth labels."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, src_path in source_assets.items():
        if src_path is not None and src_path.exists():
            dest_path = destination_dir / src_path.name
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)
            copied[key] = dest_path
    return copied


def get_peak_memory_mb(pid: int) -> float:
    """Reads peak memory on Linux, or falls back safely on Windows."""
    try:
        status_file = Path(f"/proc/{pid}/status")
        if status_file.exists():
            with open(status_file, "r") as f:
                for line in f:
                    if line.startswith("VmPeak:") or line.startswith("VmHWM:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
    except Exception:
        pass

    if resource is not None:
        try:
            ru = resource.getrusage(resource.RUSAGE_CHILDREN)
            return float(ru.ru_maxrss) / 1024.0
        except Exception:
            pass

    return 0.0


def run_solution_subprocess(
    command: List[str],
    cwd: Path,
    timeout_seconds: int = 60,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Telemetry, str, str]:
    """Executes a solution script in a subprocess with cross-platform timeout handling."""
    exec_env = os.environ.copy()
    exec_env["PYTHONPATH"] = f"{cwd}{os.pathsep}{exec_env.get('PYTHONPATH', '')}"
    if env:
        exec_env.update(env)

    start_time = time.perf_counter()
    timeout_occurred = False
    error_msg = None
    exit_code = 0
    stdout_str = ""
    stderr_str = ""
    peak_mem_mb = 0.0

    try:
        kwargs = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": exec_env,
        }

        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        proc = subprocess.Popen(command, **kwargs)

        try:
            while proc.poll() is None:
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_seconds)
                current_mem = get_peak_memory_mb(proc.pid)
                if current_mem > peak_mem_mb:
                    peak_mem_mb = current_mem
                time.sleep(0.05)

            out, err = proc.communicate()
            stdout_str = out or ""
            stderr_str = err or ""
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            timeout_occurred = True
            error_msg = f"Execution timed out after {timeout_seconds} seconds"
            try:
                proc.kill()
            except Exception:
                pass
            out, err = proc.communicate()
            stdout_str = out or ""
            stderr_str = (err or "") + f"\n[Runner] {error_msg}"
            exit_code = -1

    except Exception as e:
        error_msg = f"Subprocess launch error: {str(e)}"
        exit_code = -1
        stderr_str = error_msg
    finally:
        elapsed_seconds = time.perf_counter() - start_time

    if exit_code != 0 and not timeout_occurred and not error_msg:
        error_msg = f"Process exited with code {exit_code}: {stderr_str.strip()[-300:] if stderr_str else ''}"

    telemetry = Telemetry(
        execution_time_seconds=round(elapsed_seconds, 4),
        memory_overhead_mb=round(peak_mem_mb, 2),
        exit_code=exit_code,
        timeout_occurred=timeout_occurred,
        error_message=error_msg,
    )

    return telemetry, stdout_str, stderr_str