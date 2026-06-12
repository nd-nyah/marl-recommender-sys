import numpy as np
import torch

from marl_recommender.agents.actor_agent import ActorAgent
from marl_recommender.agents.multi_agent_controller import MultiAgentController
from marl_recommender.algorithms.mappo.actor_network import ActorNetwork


class DummyPolicy(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return self.net(x)


def test_actor_agent_act():

    obs_dim = 10
    action_dim = 5

    policy = DummyPolicy(obs_dim, action_dim)

    agent = ActorAgent(
        agent_id="movie_agent",
        policy_network=policy,
    )

    obs = np.random.randn(obs_dim)

    output = agent.act(obs)

    assert "action" in output
    assert "log_prob" in output

    assert isinstance(output["action"], int)
    assert 0 <= output["action"] < action_dim


def test_multi_agent_controller():

    obs_dim = 8
    action_dim = 4

    agents = {}

    for i in range(3):

        policy = DummyPolicy(obs_dim, action_dim)

        agents[f"agent_{i}"] = ActorAgent(
            agent_id=f"agent_{i}",
            policy_network=policy,
        )

    controller = MultiAgentController(agents)

    obs = {
        "agent_0": np.random.randn(obs_dim),
        "agent_1": np.random.randn(obs_dim),
        "agent_2": np.random.randn(obs_dim),
    }

    actions, log_probs = controller.act(obs)

    assert len(actions) == 3
    assert len(log_probs) == 3

    for k in actions:
        assert isinstance(actions[k], int)
        assert 0 <= actions[k] < action_dim