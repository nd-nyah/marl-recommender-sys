import torch

from marl_recommender.envs.recommendation_env import RecommendationEnv
from marl_recommender.agents.actor_agent import ActorAgent
from marl_recommender.agents.multi_agent_controller import MultiAgentController
from marl_recommender.evaluation.evaluator import Evaluator


# -------------------------
# Dummy Policy
# -------------------------
class DummyPolicy(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Controller Builder
# -------------------------
def build_controller(env):

    obs_dim = env.state_builder.get_state_dim()
    action_dim = len(env.items)

    agents = {}

    for agent_id in env.agents:

        policy = DummyPolicy(obs_dim, action_dim)

        agents[agent_id] = ActorAgent(
            agent_id=agent_id,
            policy_network=policy,
        )

    return MultiAgentController(agents)


# =====================================================
# TEST 1 — evaluation runs without crashing
# =====================================================
def test_evaluation_runs():

    env = RecommendationEnv(max_steps=5, mode="test")
    controller = build_controller(env)

    evaluator = Evaluator(env, controller)

    result = evaluator.run_episode()

    assert isinstance(result, dict)
    assert len(result) > 0

    # required keys (from your evaluator)
    assert "reward" in result
    assert "diversity" in result
    assert "total_reward" in result


# =====================================================
# TEST 2 — full offline evaluation loop
# =====================================================
def test_offline_evaluation():

    env = RecommendationEnv(max_steps=5, mode="test")
    controller = build_controller(env)

    results = []

    for _ in range(3):

        evaluator = Evaluator(env, controller)
        results.append(evaluator.run_episode())

    # check structure
    assert isinstance(results, list)
    assert len(results) == 3

    # each result must be a dict
    for r in results:
        assert isinstance(r, dict)
        assert "reward" in r
        assert "diversity" in r
        assert "total_reward" in r

    # aggregated sanity check
    avg_reward = sum(r["reward"] for r in results) / len(results)
    assert isinstance(avg_reward, float)