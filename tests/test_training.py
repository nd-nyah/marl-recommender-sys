import torch

from marl_recommender.envs.recommendation_env import RecommendationEnv
from marl_recommender.agents.actor_agent import ActorAgent
from marl_recommender.agents.multi_agent_controller import MultiAgentController


class DummyPolicy(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return self.net(x)


def build_test_controller(env):

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
# TEST 1
# =====================================================
def test_env_reset_and_step():

    env = RecommendationEnv(max_steps=5, mode="test")

    obs, _ = env.reset()

    controller = build_test_controller(env)

    actions, _ = controller.act(obs)

    next_obs, rewards, terminated, truncated, info = env.step(actions)

    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert len(rewards) == len(env.agents)


# =====================================================
# TEST 2
# =====================================================
def test_rollout_loop_runs():

    env = RecommendationEnv(max_steps=5, mode="test")

    controller = build_test_controller(env)

    obs, _ = env.reset()

    total_reward = 0.0
    step = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        actions, _ = controller.act(obs)

        obs, rewards, terminated, truncated, info = env.step(actions)

        # FIX: safe float conversion
        total_reward += float(list(rewards.values())[0])

        step += 1

        if step > 20:
            break

    assert isinstance(total_reward, float)
    assert step > 0