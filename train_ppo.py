import torch
import torch.nn as nn

import ray
from ray import air, tune
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

    def forward(self, input_dict, state, seq_lens):
        states = input_dict["obs"]
        action_mask = states[:, -52:]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": states})

        # Apply the mask to the logits.
        large_compatible_negative = torch.finfo(logits.dtype).min if logits.dtype == torch.float32 else 1e-9
        masked_logits = logits * action_mask + (1 - action_mask) * large_compatible_negative
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
    
if __name__ == '__main__':
    ray.shutdown()
    ray.init()

    ModelCatalog.register_custom_model("my_model", MyModel)
    register_env("my_env", lambda config: RayEnv(config))

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
            model = {
                "custom_model": "my_model",
                "custom_model_config": {
                    "fcnet_hiddens": [256, 256],
                },
            },
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
        .debugging(
            log_level="INFO",
        )
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
