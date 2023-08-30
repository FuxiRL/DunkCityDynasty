import gymnasium as gym
from DunkCityDynasty.env.base import BaseEnv

class BaseWrapper():
    def __init__(self, config):
        self.config = config
        self.observation_space = None
        self.action_space = None

    def states_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states[key] = state_infos[key][1]
        return states,infos

    def rewards_wrapper(self, states):
        rewards = {}
        for key in states.keys():
            rewards[key] = 0
        return rewards
    
class GymEnv(gym.Env):
    def __init__(self, config, wrapper=None):
        self.config = config
        self.external_env = BaseEnv(config)
        self.env_wrapper = wrapper if wrapper is not None else BaseWrapper({})
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space

    def reset(self, user_name = None, render = True):
        # env reset
        state_infos = self.external_env.reset(user_name = user_name, render = render)
        # feature embedding
        states, infos = self.env_wrapper.states_wrapper(state_infos)
        return states, infos
        
    def step(self, action_dict):
        # env reset
        state_infos, truncated, done = self.external_env.step(action_dict)

        # feature embedding
        states, infos = self.env_wrapper.states_wrapper(state_infos)
        rewards = self.env_wrapper.rewards_wrapper(state_infos)

        dones = {"__all__": done}

        return states, rewards, dones, truncated, infos
