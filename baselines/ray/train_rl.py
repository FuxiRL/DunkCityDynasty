import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import ray
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import override
from DunkCityDynasty.env.ray_env import RayEnv
from baselines.common.wrappers import RLWrapper
from baselines.common.model import GlobalStateLayer, AgentStateLayer

class MyModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        
        self.global_state_layer_dim = 64
        self.agent_state_layer_dim = 128
        self.global_state_layer = GlobalStateLayer()
        self.self_state_layer = AgentStateLayer()
        self.ally0_state_layer = AgentStateLayer()
        self.ally1_state_layer = AgentStateLayer()
        self.enemy0_state_layer = AgentStateLayer()
        self.enemy1_state_layer = AgentStateLayer()
        self.enemy2_state_layer = AgentStateLayer()
        # self.skill_layer = SkillLayer()
        self.share_layer_dim = self.global_state_layer_dim + self.agent_state_layer_dim * 6
        self.share_layer = nn.Sequential(nn.Linear(self.share_layer_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.value_layer = nn.Sequential(nn.Linear(128, 1))
        self.action_layer = nn.Sequential(nn.Linear(128, 52))
        self.value = None
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        states = input_dict["obs"]
        global_feature = states[:,:30]
        self_feature = states[:,30:103]
        ally0_feature = states[:,103:176]
        ally1_feature = states[:,176:249]
        enemy0_feature = states[:,249:322]
        enemy1_feature = states[:,322:395]
        enemy2_feature = states[:,395:468]
        action_mask = states[:, -52:]
        global_feature = self.global_state_layer(global_feature)
        self_feature = self.self_state_layer(self_feature)
        ally0_feature = self.ally0_state_layer(ally0_feature)
        ally1_feature = self.ally1_state_layer(ally1_feature)
        enemy0_feature = self.enemy0_state_layer(enemy0_feature)
        enemy1_feature = self.enemy1_state_layer(enemy1_feature)
        enemy2_feature = self.enemy2_state_layer(enemy2_feature)
        # skill_feature = self.skill_layer(action_mask)
        x = torch.cat([global_feature,self_feature, ally0_feature, ally1_feature, enemy0_feature, enemy1_feature, enemy2_feature], dim=1)
        x = self.share_layer(x.float())
        self.value = self.value_layer(x).squeeze(1)
        logits = self.action_layer(x)

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Apply the mask to the logits.
        large_compatible_negative = torch.finfo(logits.dtype).min if logits.dtype == torch.float32 else 1e-9
        masked_logits = logits * action_mask + (1 - action_mask) * large_compatible_negative

        # Return masked logits.
        return masked_logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self.value is not None, "must call forward() first"
        return self.value
     
if __name__ == '__main__':
    ray.shutdown()
    ray.init()
    wrapper = RLWrapper({'concated_states':True})
    ModelCatalog.register_custom_model("my_model", MyModel)
    register_env("my_env", lambda config: RayEnv(config, wrapper))

    env_config = {
        'id': id,
        'env_setting': 'win',
        'client_path': 'game_package_release',
        'rl_server_ip': '127.0.0.1',
        'rl_server_port': 12345,
        'game_server_ip': 'xxx.xxx.xxx.xxx',
        'game_server_port': 00000,
        'machine_server_ip': '',
        'machine_server_port': 0,
        "user_name": "username",
        'render': True,
    }

    obs_space = wrapper.observation_space
    act_space = wrapper.action_space
    config = (
        PPOConfig()
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1,
            # rollout_fragment_length='auto',
            enable_connectors=False,
            recreate_failed_workers=True,
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
            },
            train_batch_size=2048,
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
    while True:
        algo.train()

    # you can use the following code to compute actions from raw states
    # import json
    # with open('states_example.json', 'r') as f:
    #     state_infos = json.load(f)
    # states,_ = wrapper.states_wrapper(state_infos)
    # print("action",algo.compute_actions(states))