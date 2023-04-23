import gymnasium as gym

from DunkCityDynasty.env.base import BaseEnv
from DunkCityDynasty.wrapper.gym_wrapper import SimpleGymWrapper


class GymEnv(BaseEnv, gym.Env):
    def __init__(self, config):
        BaseEnv.__init__(self, config)

        self.env_wrapper = SimpleGymWrapper({})
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space

    def reset(self):
        # wrapper reset
        self.env_wrapper.reset()

        # env reset
        origin_states = BaseEnv.reset(self)

        # feature embedding
        states = self.env_wrapper.state_wrapper(origin_states)
        infos = self.env_wrapper.info_wrapper(origin_states)

        return states, infos
        
    def step(self, action_dict):
        # env reset
        origin_states, done = BaseEnv.step(self, action_dict)

        # feature embedding
        states = self.env_wrapper.state_wrapper(origin_states)
        rewards = self.env_wrapper.reward_wrapper(origin_states)
        infos = self.env_wrapper.info_wrapper(origin_states)
        dones = {"__all__": done}
        truncated = {"__all__": done}

        return states, rewards, dones, truncated, infos