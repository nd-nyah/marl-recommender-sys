from __future__ import annotations

from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from marl_recommender.envs.state_builder import StateBuilder
from marl_recommender.envs.user_simulator import UserSimulator


class RecommendationEnv(gym.Env):
    """
    Multi-agent MARL recommendation environment (CTDE + MAPPO-ready)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir="data/processed",
        max_steps=20,
        agents=("agent_0", "agent_1", "agent_2"),
        mode="real",
    ):
        super().__init__()

        # -----------------------
        # PATHS
        # -----------------------
        self.data_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
        self.mode = mode

        # -----------------------
        # DATA
        # -----------------------
        if self.mode == "real":
            self.users = pd.read_parquet(self.data_dir / "users.parquet")
            self.items = pd.read_parquet(self.data_dir / "items.parquet")

        elif self.mode in ["test", "mock"]:
            self.users = pd.read_parquet(self.data_dir / "users.parquet").head(5)
            self.items = pd.read_parquet(self.data_dir / "items.parquet").head(50)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # -----------------------
        # COMPONENTS
        # -----------------------
        self.state_builder = StateBuilder()
        self.user_simulator = UserSimulator()

        # -----------------------
        # AGENTS
        # -----------------------
        self.agents = list(agents)

        # -----------------------
        # EPISODE STATE
        # -----------------------
        self.max_steps = max_steps
        self.current_step = 0
        self.current_user = None

        # -----------------------
        # SPACES
        # -----------------------
        obs_dim = self.state_builder.get_state_dim()

        self.observation_space = spaces.Dict({
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.agents
        })

        self.action_space = spaces.Dict({
            agent: spaces.Discrete(len(self.items))
            for agent in self.agents
        })

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_user = self.users.sample(1).iloc[0]

        state = np.asarray(
            self.state_builder.build_state(self.current_user),
            dtype=np.float32,
        )

        obs = {agent: state.copy() for agent in self.agents}

        return obs, {}

    # =====================================================
    # STEP
    # =====================================================
    def step(self, actions: dict):

        chosen_items = [actions[a] for a in self.agents]

        # -----------------------
        # INDIVIDUAL REWARDS
        # -----------------------
        base_rewards = []
        for item_id in chosen_items:
            item = self.items.iloc[item_id]

            r = self.user_simulator.get_reward(
                self.current_user,
                item,
            )

            base_rewards.append(float(r))

        # -----------------------
        # METRICS
        # -----------------------
        if len(chosen_items) == 0:
            agreement = 1.0
        else:
            agreement = len(set(chosen_items)) / len(chosen_items)

        diversity = len(set(chosen_items)) / max(len(self.agents), 1)
        best_item_score = float(max(base_rewards)) if base_rewards else 0.0

        # -----------------------
        # GLOBAL REWARD
        # -----------------------
        global_reward = float(
            0.6 * best_item_score +
            0.2 * agreement +
            0.2 * diversity
        )

        rewards = {
            agent: float(global_reward)
            for agent in self.agents
        }

        # -----------------------
        # TERMINATION
        # -----------------------
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False

        # -----------------------
        # NEXT STATE
        # -----------------------
        next_state = np.asarray(
            self.state_builder.build_state(self.current_user),
            dtype=np.float32,
        )

        obs = {agent: next_state.copy() for agent in self.agents}

        # -----------------------
        # INFO
        # -----------------------
        info = {
            "chosen_items": chosen_items,
            "base_rewards": base_rewards,
            "agreement": float(agreement),
            "diversity": float(diversity),
            "global_reward": float(global_reward),
        }

        return obs, rewards, terminated, truncated, info