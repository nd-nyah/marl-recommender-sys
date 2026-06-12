import torch
import torch.nn.functional as F


def compute_returns(rewards, dones, gamma=0.99):

    returns = []
    R = 0

    for reward, done in zip(reversed(rewards), reversed(dones)):

        if done:
            R = 0

        R = reward + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


class PPOUpdate:

    def __init__(self, actors, critic, lr=3e-4, clip_eps=0.2):
        self.actors = actors
        self.critic = critic
        self.clip_eps = clip_eps

        self.actor_optim = torch.optim.Adam(
            [p for a in actors.values() for p in a.parameters()],
            lr=lr,
        )

        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

    def update(self, actors, critic, buffer, advantages, returns):

        states = torch.tensor(
            np.array(buffer.states),
            dtype=torch.float32
        )

        actions = torch.tensor(buffer.actions)

        old_log_probs = torch.stack(buffer.log_probs)

        advantages = advantages.detach()
        returns = returns.detach()

        # ----------------------
        # CRITIC UPDATE
        # ----------------------
        values = critic(states).squeeze(-1)

        value_loss = F.mse_loss(values, returns)

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # ----------------------
        # ACTOR UPDATE (PPO CLIP)
        # ----------------------
        total_actor_loss = 0.0
        idx = 0

        for agent_id, actor in actors.items():

            logits = actor(states)
            dist = torch.distributions.Categorical(logits=logits)

            log_probs = dist.log_prob(actions[:, idx])

            ratio = torch.exp(log_probs - old_log_probs[:, idx])

            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

            actor_loss = -torch.mean(
                torch.min(ratio * advantages, clipped * advantages)
            )

            total_actor_loss += actor_loss
            idx += 1

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()

        return {
            "actor_loss": total_actor_loss.item(),
            "critic_loss": value_loss.item(),
        }