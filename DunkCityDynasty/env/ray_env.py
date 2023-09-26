from ray.rllib.env import MultiAgentEnv

from DunkCityDynasty.env.base import BaseEnv

class RayEnv(MultiAgentEnv):
    def __init__(self, config, wrapper=None):
        self.config = config
        self.external_env = BaseEnv(config)
        self.env_wrapper = wrapper
        self.user_name = self.config['user_name']
        self.render_flag = self.config['render']
        self.external_env.id = self.config.worker_index
        self.external_env.rl_server_port += self.config.worker_index
        # competitors must have different user_names
        self.user_name = self.user_name  + str(self.config.worker_index)
        self.observation_space = self.env_wrapper.observation_space
        self.action_space = self.env_wrapper.action_space
        self.last_states = None


    def reset(self, **kwargs):
        # env reset
        state_infos = self.external_env.reset(user_name = self.user_name, render = self.render_flag)

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
    