import torch
import argparse, os, json

from marl_recommender.envs.recommendation_env import RecommendationEnv
from marl_recommender.evaluation.offline_evaluation import run_offline_eval
from marl_recommender.agents.multi_agent_controller import MultiAgentController
from marl_recommender.agents.actor_agent import ActorAgent
from marl_recommender.algorithms.mappo.actor_network import ActorNetwork


def build_controller(env, checkpoint_path):

    obs_dim = env.state_builder.get_state_dim()
    action_dim = len(env.items)

    agents = {}

    for agent_id in env.agents:

        policy = ActorNetwork(
            input_dim=obs_dim,
            action_dim=action_dim,
        )

        policy.eval()  # IMPORTANT for evaluation stability

        agents[agent_id] = ActorAgent(
            agent_id=agent_id,
            policy_network=policy,
        )

    return MultiAgentController(agents)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    env = RecommendationEnv(mode="test")

    controller = build_controller(env, args.checkpoint)

    results_list = []

    # -----------------------
    # run multiple episodes
    # -----------------------
    for _ in range(args.episodes):

        results = run_offline_eval(
            env,
            controller,
            episodes=1
        )

        results_list.append(results)

    # -----------------------
    # SAVE JSON
    # -----------------------
    os.makedirs("results/evaluation", exist_ok=True)

    with open("results/evaluation/results.json", "w") as f:
        json.dump(results_list, f, indent=2)

    print("\nSaved results → results/evaluation/results.json")

    # -----------------------
    # PRINT SUMMARY
    # -----------------------
    print("\n===== EVALUATION RESULTS =====")

    avg_results = {
        k: sum(r[k] for r in results_list) / len(results_list)
        for k in results_list[0].keys()
    }

    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")

# def main():

#     env = RecommendationEnv(mode="test")

#     controller = build_controller(env)

#     results = run_offline_eval(
#         env,
#         controller,
#         episodes=10
#     )

#     print("\n===== EVALUATION RESULTS =====")

#     for k, v in results.items():
#         print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

