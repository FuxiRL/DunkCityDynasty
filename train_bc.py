#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 14:59:30
LastEditor: JiangJi
LastEditTime: 2023-04-27 16:33:10
Discription: 
'''
import sys,os
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.bc import BCConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.policy import PolicySpec

from DunkCityDynasty.env.bc_env import RayBCEnv
from DunkCityDynasty.wrapper.gym_wrapper import RayBCWrapper

if __name__ == '__main__':
    ray.shutdown()
    ray.init()
    wrapper = RayBCWrapper({})
    obs_space = wrapper.observation_space
    act_space = wrapper.action_space
    register_env("my_env", lambda config: RayBCEnv(config))
    env_config = {
        'id': 1,
        'env_setting': 'win',
        'client_path': 'path-to-game-client',
        'rl_server_ip': '127.0.0.1',
        'rl_server_port': 42026,
        'game_server_ip': '127.0.0.1',
        'game_server_port': 18000,
        'episode_horizon': 100000,
    }

    
    config = (
        BCConfig()
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
        .debugging(
        log_level="INFO",
        )
        .offline_data(
        input_ = f"{os.getcwd()}\\human_data\\human_data_processed.json"
        )
    )

    algo = config.build()
    tuner = tune.Tuner(
        "BC",
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
