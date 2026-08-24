import importlib, pkgutil
from typing import Dict, List, Optional
from src.common.types import TaskDifficulty
from src.tasks.base import Task


class TaskRegistry:
    _tasks: Dict[str, Task] = {}

    @classmethod
    def register(cls, task: Task) -> Task:
        cls._tasks[task.task_id] = task
        return task

    @classmethod
    def get(cls, task_id: str) -> Task:
        cls.auto_discover()
        return cls._tasks[task_id]

    @classmethod
    def list_tasks(cls, difficulty: Optional[str] = None) -> List[Task]:
        cls.auto_discover()
        tasks = list(cls._tasks.values())
        if difficulty and difficulty.lower() != "all":
            d_enum = TaskDifficulty.from_str(difficulty)
            tasks = [t for t in tasks if t.difficulty == d_enum]
        return tasks

    @classmethod
    def auto_discover(cls) -> None:
        if cls._tasks: return
        import src.tasks.benchmark_tasks as bench
        for _, mod_name, _ in pkgutil.iter_modules(bench.__path__):
            importlib.import_module(f"src.tasks.benchmark_tasks.{mod_name}")


def register_task(task: Task) -> Task:
    return TaskRegistry.register(task)