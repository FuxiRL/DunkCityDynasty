from ray.rllib.env import MultiAgentEnv

from DunkCityDynasty.env.base import BaseEnv
from DunkCityDynasty.wrapper.gym_wrapper import SimpleWrapper


class RayEnv(MultiAgentEnv):
    def __init__(self, config):
        self.config = config
        self.external_env = BaseEnv(config)
        try:
            self.external_env.id = config.worker_index
            self.external_env.rl_server_port += config.worker_index
        except:
            print("config.worker_index not found")
            pass

        self.env_wrapper = SimpleWrapper({})
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space
        self.last_states = None


    def reset(self, **kwargs):
        # env reset
        raw_states = self.external_env.reset()

        # env reset
        states = self.env_wrapper.states_wrapper(raw_states)
        infos = self.env_wrapper.infos_wrapper(raw_states)

        self.last_states = states

        return states, infos
    
    def step(self, action_dict):
        # env reset
        try: # avoid game disconnect error
            raw_states, done = self.external_env.step(action_dict)
            # feature embedding
            states = self.env_wrapper.states_wrapper(raw_states)
            self.last_states = states
            rewards = self.env_wrapper.reward_wrapper(raw_states)
            infos = self.env_wrapper.infos_wrapper(raw_states)
            dones = {"__all__": done}
            truncated = {"__all__": done}
        except:
            states = self.last_states
            rewards = {key: 0 for key in states.keys()}
            dones = {"__all__": True}
            infos = {key: {} for key in states.keys()}
            truncated = {"__all__": True}

        return states, rewards, dones, truncated, infos
    