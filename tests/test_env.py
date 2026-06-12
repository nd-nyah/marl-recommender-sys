import numpy as np
from marl_recommender.envs.recommendation_env import RecommendationEnv


def test_environment():

    env = RecommendationEnv(mode="test")

    # -------------------------
    # RESET
    # -------------------------
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    assert set(obs.keys()) == set(env.agents)

    state_dim = env.state_builder.get_state_dim()

    for agent in env.agents:
        assert isinstance(obs[agent], np.ndarray)
        assert obs[agent].shape == (state_dim,)
        assert obs[agent].dtype == np.float32

    # -------------------------
    # SAMPLE ACTIONS
    # -------------------------
    action = {
        agent: env.action_space.spaces[agent].sample()
        for agent in env.agents
    }

    # -------------------------
    # STEP
    # -------------------------
    next_obs, reward, terminated, truncated, info = env.step(action)

    # -------------------------
    # OBS CHECK
    # -------------------------
    assert isinstance(next_obs, dict)
    assert set(next_obs.keys()) == set(env.agents)

    for agent in env.agents:
        assert isinstance(next_obs[agent], np.ndarray)
        assert next_obs[agent].shape == (state_dim,)
        assert next_obs[agent].dtype == np.float32

    # -------------------------
    # REWARD CHECK
    # -------------------------
    assert isinstance(reward, dict)
    assert set(reward.keys()) == set(env.agents)

    for r in reward.values():
        assert isinstance(r, float)

    # -------------------------
    # DONE FLAGS
    # -------------------------
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # -------------------------
    # INFO CHECK
    # -------------------------
    assert isinstance(info, dict)

    required_keys = {
        "chosen_items",
        "base_rewards",
        "agreement",
        "diversity",
        "global_reward",
    }

    for key in required_keys:
        assert key in info

    assert isinstance(info["chosen_items"], list)
    assert isinstance(info["base_rewards"], list)
    assert isinstance(info["agreement"], float)
    assert isinstance(info["diversity"], float)
    assert isinstance(info["global_reward"], float)