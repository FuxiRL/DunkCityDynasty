from ray.rllib.env import MultiAgentEnv

from DunkCityDynasty.env.base import BaseEnv
from DunkCityDynasty.wrapper.bc_wrapper import RayBCWrapper


class RayBCEnv(MultiAgentEnv):
    def __init__(self, config):
        self.config = config
        self.external_env = BaseEnv(config)
        try:
            self.external_env.id = config.worker_index
            self.external_env.rl_server_port += config.worker_index
        except:
            print("config.worker_index not found")
            pass

        self.env_wrapper = RayBCWrapper({})
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space

    def reset(self, **kwargs):
        # wrapper reset
        self.env_wrapper.reset()

        # env reset
        raw_states = self.external_env.reset()

        # env reset
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
    