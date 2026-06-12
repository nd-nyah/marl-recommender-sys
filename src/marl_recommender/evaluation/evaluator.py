from marl_recommender.metrics.tracker import MetricTracker
from marl_recommender.metrics.diversity import intra_list_diversity


class Evaluator:

    def __init__(self, env, controller):
        self.env = env
        self.controller = controller
        self.tracker = MetricTracker()

    def run_episode(self):

        obs, _ = self.env.reset()

        terminated = False
        truncated = False

        all_items = []
        reward_history = []

        while not (terminated or truncated):

            actions, _ = self.controller.act(obs)

            obs, rewards, terminated, truncated, info = self.env.step(actions)

            reward = sum(rewards.values())

            reward_history.append(reward)
            all_items.extend(info["chosen_items"])

            self.tracker.log("reward", reward)

        # -------------------------
        # FINAL EPISODE METRICS
        # -------------------------
        self.tracker.log(
            "diversity",
            intra_list_diversity(all_items)
        )

        self.tracker.log(
            "total_reward",
            sum(reward_history)
        )

        return self.tracker.summary()