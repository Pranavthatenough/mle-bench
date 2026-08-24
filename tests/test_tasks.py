import unittest
from src.tasks.registry import TaskRegistry


class TestTasks(unittest.TestCase):
    def test_task_registry(self):
        tasks = TaskRegistry.list_tasks()
        self.assertGreaterEqual(len(tasks), 6)

    def test_get_single_task(self):
        task = TaskRegistry.get("task_low_01")
        self.assertEqual(task.task_id, "task_low_01")
        self.assertIsNotNone(task.dataset_path)


if __name__ == "__main__":
    unittest.main()