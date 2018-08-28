import numpy as np

from gym.spaces import Box

from .basic_engagement_gym_base import BasicEngagementGymBase


class BasicEngagementGymCoach(BasicEngagementGymBase):
    def get_observation_space(self):
        # time              -> 4 values
        # day               -> 2 values
        # location          -> 3 values
        # activity          -> 5 values
        # last notificatoin -> 2 values
        return Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([3, 1, 2, 4, 1]),
        )

    def intepret_state(self, state_tuple):
        return np.array(state_tuple)
