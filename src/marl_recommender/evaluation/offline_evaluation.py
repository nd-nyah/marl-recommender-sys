from marl_recommender.evaluation.evaluator import Evaluator


def run_offline_eval(env, controller, episodes=10):

    results = []

    for _ in range(episodes):
        evaluator = Evaluator(env, controller)
        results.append(evaluator.run_episode())

    final = {}

    for k in results[0].keys():
        final[k] = sum(r[k] for r in results) / len(results)

    return final