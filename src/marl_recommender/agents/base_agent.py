# src/marl_recommender/agents/base_agent.py

from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self, agent_id):
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation):
        pass