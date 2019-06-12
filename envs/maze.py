"""Gym environment wrapper for 2D maze grid world."""

import copy
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from pycolab import rendering

from .pycolab_envs import maze


class GridNavigationEnv(gym.Env):
    """Gym environment wrapper for 2D pycolab grid navigation environment."""

    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.layers = ('#', 'P')
        self.num_actions = 4

        self.np_random = None
        self.game = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()
        self.reset()

    def _obs_to_np_array(self, obs):
        return copy.copy(self.renderer(obs))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.game = maze.make_game()
        obs, _, _ = self.game.its_showtime()
        return self._obs_to_np_array(obs)

    def step(self, action):
        obs, _, reward = self.game.play(action)
        return self._obs_to_np_array(obs), reward, False, self.game.the_plot
