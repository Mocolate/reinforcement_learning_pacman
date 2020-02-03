import pytest

from qlearningAgents import QLearningAgent


def test_get_q_value_returns_zero_for_unset_state_action():
    agent = QLearningAgent()

    assert 0.0 == agent.getQValue(state="S", action=0)
    assert 0.0 == agent.getQValue(state=1, action="a")


def test_get_q_value_returns_the_value_that_was_set():
    agent = QLearningAgent()
    agent._setQValue(state="S", action=0, value=1.0)
    agent._setQValue(state=1, action="a", value=-1.0)

    assert 1.0 == agent.getQValue(state="S", action=0)
    assert -1.0 == agent.getQValue(state=1, action="a")


# def test_set_q_value_does_not_accept_anything_else_than_float():
#     agent = QLearningAgent()
#     with pytest.raises(FloatingPointError):
#         agent._setQValue(state="S", action=0, value="Not Allowed")
