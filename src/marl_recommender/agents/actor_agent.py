import torch
from torch.distributions import Categorical

from marl_recommender.agents.base_agent import BaseAgent


class ActorAgent(BaseAgent):

    def __init__(
        self,
        agent_id: str,
        policy_network,
        device: str = "cpu",
    ):
        super().__init__(agent_id)

        self.policy = policy_network
        self.device = device

    def act(self, observation):

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(observation)

        dist = Categorical(logits=logits)

        action = dist.sample()

        return {
            "action": action.item(),
            "log_prob": dist.log_prob(action),
        }

    def evaluate_actions(self, observations, actions):

        logits = self.policy(observations)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        return log_probs, entropy