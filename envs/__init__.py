from gym.envs.registration import register


register(
    'GridNavigation-v0',
    entry_point='envs.maze:GridNavigationEnv',
    max_episode_steps=5000
)
