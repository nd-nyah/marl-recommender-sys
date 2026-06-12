"""
Simple user simulator for MARL recommender.
"""

from __future__ import annotations

import numpy as np


class UserSimulator:
    """
    Simulates user feedback.

    Reward:
        1 if item popularity is above user's
        positive rate threshold.
    """

    def __init__(self):
        pass

    def get_reward(
        self,
        user_features,
        item_features,
    ):
        popularity = float(
            item_features.get(
                "popularity",
                0.0,
            )
        )

        positive_rate = float(
            user_features.get(
                "positive_rate",
                0.5,
            )
        )

        reward = (
            1.0
            if popularity > positive_rate
            else 0.0
        )

        return reward

    def is_positive(
        self,
        reward,
    ):
        return reward > 0.0