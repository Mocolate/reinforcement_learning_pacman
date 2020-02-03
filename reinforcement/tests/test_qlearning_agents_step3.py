import pytest

from qlearningAgents import QLearningAgent


class AgentTest(QLearningAgent):
    def __init__(self, legal_actions):
        QLearningAgent.__init__(self)
        self.legal_actions = legal_actions
        for state in legal_actions:
            for index in range(0, len(legal_actions[state][0])):
                self._setQValue(state=state, action=legal_actions[state][0][index], value=legal_actions[state][1][index])

    def getLegalActions(self, state):
        return self.legal_actions[state][0]


def test_when_there_are_no_legal_actions_then_compute_action_from_q_values_returns_none():
    agent = AgentTest({"state1": ([], [])})

    assert agent.computeActionFromQValues("state1") is None


def test_when_there_are_legal_actions_then_compute_value_from_q_values_returns_the_action_with_the_highest_q_value():
    agent = AgentTest({"state1": ([1, 2, 3], [-1.0, -2.0, -3.0])})

    assert 1 == agent.computeActionFromQValues("state1")


def test_when_there_are_legal_actions_with_equal_best_q_values_then_compute_value_from_q_values_returns_random_best_action():
    agent = AgentTest({"state1": ([1, 2, 3], [-1.0, -2.0, -1.0])})

    results = []
    for index in range(0, 10000):
        results.append(agent.computeActionFromQValues("state1"))

    assert 1 in results
    assert 3 in results
