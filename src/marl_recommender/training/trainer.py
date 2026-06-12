import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, controller, algorithm):
        self.env = env
        self.controller = controller
        self.algorithm = algorithm
        
        # FIX 1: initialize writer
        self.writer = SummaryWriter("logs/tensorboard/trainer")

        # FIX 2: episode counter
        self.episode = 0

    def collect_episode(self):

        obs, _ = self.env.reset()

        done = False
        total_reward = 0.0

        buffer = self.algorithm.buffer

        while not done:

            # -----------------------
            # DECENTRALIZED ACTORS
            # -----------------------
            actions, log_probs = self.controller.act(obs)

            # -----------------------
            # CENTRALIZED STATE (CTDE FIX)
            # -----------------------
            state = self.env.state_builder.build_state(
                self.env.current_user
            )

            state = torch.tensor(state, dtype=torch.float32)

            # -----------------------
            # ENV STEP
            # -----------------------
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)

            reward = float(list(rewards.values())[0])
            done = bool(terminated or truncated)

            total_reward += reward

            # -----------------------
            # STORE CTDE TRANSITION
            # -----------------------
            buffer.store(
                state=state.numpy(),
                actions=actions,
                log_probs=log_probs,
                reward=reward,
                done=done,
            )

            obs = next_obs

        # logging
        self.writer.add_scalar(
            "reward/episode",
            total_reward,
            self.episode
        )

        self.episode += 1

        return total_reward
