from __future__ import annotations

import numpy as np
import pandas as pd


class StateBuilder:
    """
    Builds user state representations for MARL recommender.

    Required user features:
        - avg_rating
        - interaction_count
        - positive_rate
    """

    def __init__(self):
        self.state_dim = 3

    def build_state(self, user_row: pd.Series | dict):
        """
        Convert a user row into a fixed-size state vector.
        """

        # -----------------------
        # VALIDATION (fail fast)
        # -----------------------
        required = {"avg_rating", "interaction_count", "positive_rate"}
        missing = required - set(user_row.keys())

        if missing:
            raise ValueError(f"Missing features: {missing}")

        # -----------------------
        # STATE VECTOR
        # -----------------------
        return np.array(
            [
                float(user_row["avg_rating"]),
                float(user_row["interaction_count"]),
                float(user_row["positive_rate"]),
            ],
            dtype=np.float32,
        )

    def get_state_dim(self):
        return self.state_dim

# from __future__ import annotations

# import numpy as np
# """
# State builder for MARL recommender environment.
# """

# class StateBuilder:
#     """
#     Builds user state representations.

#     Current state:
#         [avg_rating,
#          interaction_count,
#          positive_rate]
#     """

#     def __init__(self):
#         self.state_dim = 3

#     def build_state(self, user_row):
#         return np.array(
#             [
#                 user_row["avg_rating"],
#                 user_row["interaction_count"],
#                 user_row["positive_rate"],
#             ],
#             dtype=np.float32,
#         )

#     def get_state_dim(self):
#         return self.state_dim