#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-24 17:13:35
LastEditor: JiangJi
LastEditTime: 2023-04-25 19:47:57
Discription: 
'''
#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-24 17:13:35
LastEditor: JiangJi
LastEditTime: 2023-04-25 19:33:55
Discription: 
'''
import torch
import torch.nn as nn
import ray
from ray import air, tune
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.policy import PolicySpec

from DunkCityDynasty.env.ray_env import RayEnv
from DunkCityDynasty.wrapper.gym_wrapper import SimpleGymWrapper


class MyModel(TorchModelV2,nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        self._last_batch_size = None

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        states = input_dict["obs"]
        action_mask = states[:, -52:]
        # action_mask = input_dict["obs"]["action_mask"]
        
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": states})
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return masked_logits, state


    def value_function(self):
        return  self.internal_model.value_function()
if __name__ == '__main__':
    ray.shutdown()
    ray.init()
    ModelCatalog.register_custom_model("my_model", MyModel)
    register_env("my_env", lambda config: RayEnv(config))
    env_config = {
        'id': 1,
        'env_setting': 'win',
        'client_path': 'E:/L33/game_package_opensource',
        'rl_server_ip': '127.0.0.1',
        'rl_server_port': 42026,
        'game_server_ip': '42.186.153.157',
        'game_server_port': 18000,
        'episode_horizon': 100000,
    }
    wrapper = SimpleGymWrapper({})
    obs_space = wrapper.observation_space
    act_space = wrapper.action_space
    config = (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length='auto',
            enable_connectors=False,
        )
        .environment(
        env = "my_env",
        env_config = env_config,
        observation_space = obs_space,
        action_space = act_space,
        )
        .training(
        train_batch_size=256,
        lr=0.0001,
        )
        .multi_agent(
            policies = {
                "default_policy": PolicySpec(observation_space=obs_space,action_space=act_space,config={}),
                "shared_policy": PolicySpec(observation_space=obs_space,action_space=act_space,config={}),
            },
            policy_mapping_fn= lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        # .offline_data(
        #     input_=_input
        # )   
    )
    algo = config.build()
    

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"episode_reward_mean": 150},
        ),
        param_space=config,
    )

    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint

    
    # test for compute_single_action

    # env = RayEnv(env_config)
    # states , _ = env.reset()
    # for i in range(1000):
    #     action_dict = {}
    #     for key in states:
    #         action_dict[key] = algo.compute_single_action(states[key])
    #     states, _, dones, _ , _ = env.step(action_dict)




