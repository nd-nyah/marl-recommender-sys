import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time

from marl_recommender.algorithms.mappo.buffer import RolloutBuffer


class MAPPO:

    def __init__(self, actors, critic, gamma=0.99, lam=0.95, lr=3e-4):
        self.actors = actors
        self.critic = critic

        self.buffer = RolloutBuffer()
        
        
        os.makedirs("logs/tensorboard", exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=f"logs/tensorboard/mappo_{int(time.time())}"
        )

        self.step = 0

        self.gamma = gamma
        self.lam = lam

        self.actor_optim = torch.optim.Adam(
            [p for a in actors.values() for p in a.parameters()],
            lr=lr,
        )

        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

    def compute_returns(self, rewards, dones, values):

        returns = []
        G = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                G = 0

            G = rewards[t] + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self):

        if len(self.buffer) == 0:
            return

        states = torch.tensor(
            np.array(self.buffer.states),
            dtype=torch.float32
        )

        agent_ids = list(self.actors.keys())

        actions = {
            a: torch.tensor(
                [step[a] for step in self.buffer.actions],
                dtype=torch.long
            )
            for a in agent_ids
        }

        old_log_probs = {
            a: torch.stack([
                step[a] for step in self.buffer.log_probs
            ])
            for a in agent_ids
        }

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)

        # -------------------------
        # CRITIC
        # -------------------------
        values = self.critic(states).squeeze(-1)

        returns = self.compute_returns(
            self.buffer.rewards,
            self.buffer.dones,
            values.detach().tolist()
        )

        returns = returns.detach()
        advantages = returns - values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------------------------
        # ACTOR UPDATE
        # -------------------------
        total_actor_loss = 0.0

        for agent_id in agent_ids:

            actor = self.actors[agent_id]

            agent_actions = actions[agent_id]
            agent_old_log_probs = old_log_probs[agent_id]

            logits = actor(states)
            dist = torch.distributions.Categorical(logits=logits)

            log_probs = dist.log_prob(agent_actions)

            ratio = torch.exp(log_probs - agent_old_log_probs)

            clipped = torch.clamp(
                ratio,
                1 - 0.2,
                1 + 0.2
            )

            actor_loss = -torch.mean(
                torch.min(ratio * advantages, clipped * advantages)
            )

            total_actor_loss += actor_loss

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()

        # -------------------------
        # CRITIC UPDATE
        # -------------------------
        value_loss = F.mse_loss(values, returns)

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # -------------------------
        # LOGGING
        # -------------------------
        self.writer.add_scalar("loss/actor", total_actor_loss.item(), self.step)
        self.writer.add_scalar("loss/critic", value_loss.item(), self.step)

        self.step += 1

        # -------------------------
        # CLEAR BUFFER
        # -------------------------
        self.buffer.clear()

    def save(self, path):
        torch.save({
            "actors": {k: v.state_dict() for k, v in self.actors.items()},
            "critic": self.critic.state_dict(),
        }, path)


## no learning - testing ok
# from marl_recommender.algorithms.mappo.buffer import (
#     RolloutBuffer,
# )

# class MAPPO:

#     def __init__(
#         self,
#         actors,
#         critic,
#     ):
#         self.actors = actors
#         self.critic = critic

#         self.buffer = RolloutBuffer()
    
#     def update(self):
#         pass

#     def save(self, path):
#         pass