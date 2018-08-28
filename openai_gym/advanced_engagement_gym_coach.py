import math
import sys
import os
import numpy as np

from gym.spaces import Box

from .advanced_engagement_gym_base import AdvancedEngagementGymBase

from human_modeling_utils import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Nurture', 'server', 'notification'))
from nurture.learning.state import State


class AdvancedEngagementGymCoach(AdvancedEngagementGymBase):
    
    def get_observation_space(self):
        return Box(
                low=np.array([
                    0.,  # time of day
                    0.,  # day of week
                    0.,  # motion - stationary
                    0.,  # motion - walking
                    0.,  # motion - running
                    0.,  # motion - driving
                    0.,  # motion - biking
                    0.,  # location - home
                    0.,  # location - work
                    0.,  # location - other
                    0.,  # notificatoin time
                    0.,  # ringer mode - silent
                    0.,  # ringer mode - vibrate
                    0.,  # ringer mode - normal
                    0.,  # screen status
                ]),
                high=np.array([
                    1.,  # time of day
                    1.,  # day of week
                    1.,  # motion - stationary
                    1.,  # motion - walking
                    1.,  # motion - running
                    1.,  # motion - driving
                    1.,  # motion - biking
                    1.,  # location - home
                    1.,  # location - work
                    1.,  # location - other
                    math.log(60.),  # notificatoin time
                    1.,  # ringer mode - silent
                    1.,  # ringer mode - vibrate
                    1.,  # ringer mode - normal
                    1.,  # screen status
                ]),
        )

    def intepret_state(self, state):
        return np.array([
            state.timeOfDay,
            state.dayOfWeek,
            state.motion == State.MOTION_STATIONARY,
            state.motion == State.MOTION_WALKING,
            state.motion == State.MOTION_RUNNING,
            state.motion == State.MOTION_DRIVING,
            state.motion == State.MOTION_BIKING,
            state.location == State.LOCATION_HOME,
            state.location == State.LOCATION_WORK,
            state.location == State.LOCATION_OTHER,
            math.log(utils.clip(state.notificationTimeElapsed, 5.0, 60.0)),
            state.ringerMode == State.RINGER_MODE_SILENT,
            state.ringerMode == State.RINGER_MODE_VIBRATE,
            state.ringerMode == State.RINGER_MODE_NORMAL,
            state.screenStatus,
        ])

