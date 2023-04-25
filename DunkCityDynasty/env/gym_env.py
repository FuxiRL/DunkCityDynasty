import gymnasium as gym

from DunkCityDynasty.env.base import BaseEnv
from DunkCityDynasty.wrapper.gym_wrapper import SimpleGymWrapper


class GymEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.external_env = BaseEnv(config)

        self.env_wrapper = SimpleGymWrapper({})
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space

    def reset(self):
        # wrapper reset
        self.env_wrapper.reset()

        # env reset
        raw_states = self.external_env.reset()

        # feature embedding
        states = self.env_wrapper.state_wrapper(raw_states)
        infos = self.env_wrapper.info_wrapper(raw_states)

        return states, infos
        
    def step(self, action_dict):
        # env reset
        raw_states, done = self.external_env.step(action_dict)

        # feature embedding
        states = self.env_wrapper.state_wrapper(raw_states)
        rewards = self.env_wrapper.reward_wrapper(raw_states)
        infos = self.env_wrapper.info_wrapper(raw_states)
        dones = {"__all__": done}
        truncated = {"__all__": done}

        return states, rewards, dones, truncated, infos