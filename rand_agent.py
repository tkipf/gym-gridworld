"""Simple random agent.

Running this script directly executes the random agent in envirnoment and stores
experience in a replay buffer.
"""

import argparse

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='GridNavigation-v0',
                        help='Select the environment to run.')
    parser.add_argument('fname', nargs='?', default='results/replay_buffer.h5',
                        help='Save path for replay buffer.')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 1
    reward = 0
    done = False

    replay_buffer = {
        'observations': [],
        'actions': []
    }

    for i in range(episode_count):
        ob = env.reset()
        replay_buffer['observations'].append(ob)

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            replay_buffer['actions'].append(action)
            replay_buffer['observations'].append(ob)

            if done:
                break

    env.close()

    # Save replay buffer to disk.
    utils.save_dict_h5py(replay_buffer, args.fname)
