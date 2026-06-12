class MultiAgentController:

    def __init__(self, agents):
        self.agents = agents

    def act(self, observations):

        actions = {}
        log_probs = {}

        for agent_id, agent in self.agents.items():

            output = agent.act(
                observations[agent_id]
            )

            actions[agent_id] = output["action"]

            log_probs[agent_id] = output["log_prob"]

        return actions, log_probs