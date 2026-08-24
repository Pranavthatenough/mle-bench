import unittest
from src.agents.loader import load_agent
from src.agents.base import BaseAgent


class TestAgents(unittest.TestCase):
    def test_load_dummy(self):
        agent = load_agent("dummy")
        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.name, "DummyMLAgent")

    def test_load_heuristic(self):
        agent = load_agent("heuristic")
        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.name, "HeuristicMLAgent")


if __name__ == "__main__":
    unittest.main()