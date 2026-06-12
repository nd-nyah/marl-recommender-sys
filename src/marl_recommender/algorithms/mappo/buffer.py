class RolloutBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    # -----------------------------------
    # STORE TRANSITION
    # -----------------------------------
    def store(self, state, actions, log_probs, reward, done):

        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.dones.append(done)

    # -----------------------------------
    # LENGTH
    # -----------------------------------
    def __len__(self):
        return len(self.states)

    # -----------------------------------
    # CLEAR BUFFER
    # -----------------------------------
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    # -----------------------------------
    # CONVERT TO TENSORS (CTDE SAFE)
    # -----------------------------------
    def to_tensors(self, agent_ids):

        import torch
        import numpy as np

        # -----------------------
        # STATES
        # -----------------------
        states = torch.tensor(
            np.array(self.states),
            dtype=torch.float32
        )

        # -----------------------
        # REWARDS / DONES
        # -----------------------
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # -----------------------
        # PER-AGENT ACTIONS / LOGPROBS
        # -----------------------
        actions = {
            a: torch.tensor(
                [step[a] for step in self.actions],
                dtype=torch.long
            )
            for a in agent_ids
        }

        log_probs = {
            a: torch.stack(
                [step[a] for step in self.log_probs]
            )
            for a in agent_ids
        }

        return states, actions, log_probs, rewards, dones


# class RolloutBuffer:

#     def __init__(self):
#         self.steps = []

#     # -----------------------------------
#     # STORE ONE TIMESTEP (JOINT TRANSITION)
#     # -----------------------------------
#     def store(self, obs, actions, log_probs, reward, done):

#         self.steps.append({
#             "obs": obs,                 # dict: agent_id -> obs
#             "actions": actions,         # dict: agent_id -> action
#             "log_probs": log_probs,     # dict: agent_id -> log_prob
#             "reward": reward,           # scalar global reward
#             "done": done,
#         })

#     # -----------------------------------
#     # CONVERT TO TENSORS FOR TRAINING
#     # -----------------------------------
#     def to_tensors(self, agent_ids):

#         import torch
#         import numpy as np

#         states = []
#         actions = {a: [] for a in agent_ids}
#         log_probs = {a: [] for a in agent_ids}
#         rewards = []
#         dones = []

#         for step in self.steps:

#             # global state = concat or shared state
#             # (you currently use shared observation)
#             any_agent = list(step["obs"].values())[0]
#             states.append(any_agent)

#             rewards.append(step["reward"])
#             dones.append(step["done"])

#             for a in agent_ids:
#                 actions[a].append(step["actions"][a])
#                 log_probs[a].append(step["log_probs"][a])

#         # tensors
#         states = torch.tensor(np.array(states), dtype=torch.float32)

#         actions = {
#             a: torch.tensor(actions[a]) for a in agent_ids
#         }

#         log_probs = {
#             a: torch.stack(log_probs[a]) for a in agent_ids
#         }

#         rewards = torch.tensor(rewards, dtype=torch.float32)
#         dones = torch.tensor(dones, dtype=torch.float32)

#         return states, actions, log_probs, rewards, dones

#     def clear(self):
#         self.steps = []