# scripts/train.py

import argparse
from pathlib import Path

from marl_recommender.envs.recommendation_env import RecommendationEnv
from marl_recommender.algorithms.mappo.critic_network import CriticNetwork
from marl_recommender.training.trainer import Trainer
from marl_recommender.agents.multi_agent_controller import MultiAgentController
from marl_recommender.agents.actor_agent import ActorAgent
from marl_recommender.algorithms.mappo.actor_network import ActorNetwork
from marl_recommender.algorithms.mappo.mappo import MAPPO
from marl_recommender.utils.config import load_config
from marl_recommender.utils.seed import set_seed


def build_controller(env):

    obs_dim = env.state_builder.get_state_dim()
    action_dim = len(env.items)

    agents = {}

    for agent_id in env.agents:

        policy = ActorNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
        )

        agents[agent_id] = ActorAgent(
            agent_id=agent_id,
            policy_network=policy,
        )

    return MultiAgentController(agents), obs_dim

def build_algorithm(controller, cfg, obs_dim):

    # import torch
    from marl_recommender.algorithms.mappo.critic_network import CriticNetwork

    # obs_dim = controller.agents[next(iter(controller.agents))].policy_network.model[0].in_features
    # obs_dim = env.state_builder.get_state_dim()
    
    actors = {
        agent_id: agent.policy
        for agent_id, agent in controller.agents.items()
    }

    critic = CriticNetwork(state_dim=obs_dim)

    return MAPPO(
        actors=actors,
        critic=critic,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 42))

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    env = RecommendationEnv(mode="real")

    controller, obs_dim  = build_controller(env)

    algorithm = build_algorithm(controller, cfg, obs_dim)

    trainer = Trainer(
        env=env,
        controller=controller,
        algorithm=algorithm,
    )

    print("Training started...\n")

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):

        reward = trainer.collect_episode()

        # keep as-is (no assumptions about internal implementation)
        algorithm.update()

        print(f"Epoch {epoch+1}/{epochs} | Reward: {reward:.4f}")

        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:

            path = f"checkpoints/mappo_epoch_{epoch+1}.pt"
            algorithm.save(path)

            print(f"Saved checkpoint → {path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()