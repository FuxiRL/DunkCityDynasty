import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import os

# def get_output_size_with_batch(model, input_size, dtype=torch.float):
#     with torch.no_grad():
#         output = model(torch.zeros([1, *input_size[1:]], dtype=dtype))
#         output_size = [None] + list(output.size())[1:]
#     return output_size

def embedding_layer(input_size, num_embeddings, embedding_dim, **kwargs):
    class EmbeddingLayer(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super(EmbeddingLayer, self).__init__()
            self.layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, **kwargs)

        def forward(self, x: torch.Tensor):
            # if x.dtype != torch.int:
            #     x = x.int()
            return self.layer(x)
    layer = EmbeddingLayer(num_embeddings, embedding_dim, **kwargs)
    output_size = [None, embedding_dim]
    # output_size = get_output_size_with_batch(layer, input_size=input_size, dtype=torch.long)
    return layer, output_size

def linear_layer(input_size, layer_dim):
    input_dim = input_size[1]
    output_size = [None, layer_dim]
    layer = nn.Sequential(nn.Linear(input_dim, layer_dim), nn.ReLU())
    return layer, output_size
class GlobalStateLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_state_len = 30
        self.linear_layer, self.linear_layer_out_dim = linear_layer([None, self.global_state_len], 64)
    def forward(self, x):
        x = x.float()
        x = self.linear_layer(x)
        return x
class AgentStateLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent_state_len = 73
        self.my_character_type_embed_layer, self.my_character_type_embed_layer_out_dim = embedding_layer([None], 100, 16)
        self.my_role_type_embed_layer, self.my_role_type_embed_layer_out_dim = embedding_layer([None], 8 ,8)
        self.my_buff_type_embed_layer, self.my_buff_type_embed_layer_out_dim = embedding_layer([None], 50, 6)
        self.agent_state_dim = 16+8+6-3 + self.agent_state_len
        self.out_dim = 128
        self.linear_layer, self.linear_layer_out_dim = linear_layer([None, self.agent_state_dim], self.out_dim )
    def forward(self, x):
        my_character_type = x[:, 0].long()
        my_role_type = x[:, 1].long()
        my_buff_type = x[:, 2].long()
        my_character_type = self.my_character_type_embed_layer(my_character_type)
        my_role_type = self.my_role_type_embed_layer(my_role_type)
        my_buff_type = self.my_buff_type_embed_layer(my_buff_type)
        my_states = x[:,3:].float()
        x = torch.cat([my_character_type, my_role_type, my_buff_type, my_states], dim=1).float()
        x = self.linear_layer(x)
        return x

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, states):
        global_feature = states[0].float()
        self_feature = states[1]
        ally0_feature = states[2]
        ally1_feature = states[3]
        enemy0_feature = states[4]
        enemy1_feature = states[5]
        enemy2_feature = states[6]
        # action_mask = states[7]
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
        value = self.value_layer(x)
        logits_p = self.action_layer(x)
        # large_negative = torch.finfo(logits_p.dtype).min if logits_p.dtype == torch.float32 else 1e-9
        # mask_logits_p = logits_p * action_mask + (1 - action_mask) * large_negative
        probs = nn.functional.softmax(logits_p, dim=1)
        return value.float(), probs.float()
class BaseWrapper():
    def __init__(self, config):
        self.config = config
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(226,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(52)
    def state_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states_dict = state_infos[key][1]
            global_state = np.array(list(states_dict['global_state'].values()))
            self_state = np.array(list(states_dict['self_state'].values()))
            ally0_state = np.array(list(states_dict['ally_0_state'].values()))
            ally1_state = np.array(list(states_dict['ally_1_state'].values()))
            enemy0_state = np.array(list(states_dict['enemy_0_state'].values()))
            enemy1_state = np.array(list(states_dict['enemy_1_state'].values()))
            enemy2_state = np.array(list(states_dict['enemy_2_state'].values()))
            action_mask = np.array(state_infos[key][-1])
            observations = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask]
            states[key] = observations
        return states,infos
    def reward_wrapper(self, states):
        return 0
    def info_wrapper(self, states):
        return {}
    def truncated_wrapper(self, states):
        return False
def onehot(num, size):
    """
    """
    onehot_vec = np.zeros(size)
    onehot_vec[num] = 1
    return onehot_vec.tolist()
class SimpleWrapper(BaseWrapper):
    ''' Simple Wrapper for Baseline
    '''
    def __init__(self, config):
        super().__init__(config)

    def state_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states_dict = state_infos[key][1]
            global_state = self.handle_global_states(states_dict['global_state'])
            self_state = self.handle_agent_states(states_dict['self_state'])
            ally0_state = self.handle_agent_states(states_dict['ally_0_state'])
            ally1_state = self.handle_agent_states(states_dict['ally_1_state'])
            enemy0_state = self.handle_agent_states(states_dict['enemy_0_state'])
            enemy1_state = self.handle_agent_states(states_dict['enemy_1_state'])
            enemy2_state = self.handle_agent_states(states_dict['enemy_2_state'])

            action_mask = np.array(state_infos[key][-1])
            observations = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask]
            states[key] = observations
        return states,infos
    def handle_global_states(self, global_states_dict):
        global_states_list = []
        global_states_list.append(global_states_dict['attack_remain_time'] * 0.05)
        global_states_list.append(global_states_dict['match_remain_time'] * 0.02)
        global_states_list.append(global_states_dict['is_home_team'])
        global_states_list.append(global_states_dict['ball_position_x']*0.2)
        global_states_list.append(global_states_dict['ball_position_y']*0.5)
        global_states_list.append(global_states_dict['ball_position_z']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_x']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_y']*0.5)
        global_states_list.append(global_states_dict['vec_ball_basket_z']*0.2)
        global_states_list.append(global_states_dict['team_own_ball'])
        global_states_list.append(global_states_dict['enemy_team_own_ball'])
        global_states_list.append(global_states_dict['ball_clear'])
        ball_status = onehot(int(global_states_dict['ball_status']), 6)
        global_states_list += ball_status
        global_states_list.append(global_states_dict['can_rebound'])
        global_states_list.append(global_states_dict['dis_to_rebound_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_y']*0.2)
        global_states_list.append(global_states_dict['can_block'])
        global_states_list.append(global_states_dict['shoot_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['shoot_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_y']*0.2)
        global_states_list.append(global_states_dict['block_diff_angle']*0.3)
        global_states_list.append(global_states_dict['block_diff_r']*0.2)
        return np.array(global_states_list)
    def handle_agent_states(self, agent_states_dict):
        agent_states_list = []
        agent_states_list.append(agent_states_dict['character_id'])
        agent_states_list.append(agent_states_dict['position_type'])
        agent_states_list.append(agent_states_dict['buff_key'])
        agent_states_list.append(agent_states_dict['buff_value']*0.1)
        agent_states_list.append((agent_states_dict['stature']-180)*0.1)
        agent_states_list.append(agent_states_dict['rational_shoot_distance']-7)
        agent_states_list.append(agent_states_dict['position_x']*0.2)
        agent_states_list.append(agent_states_dict['position_y']*0.5)
        agent_states_list.append(agent_states_dict['position_z']*0.2)
        agent_states_list.append(agent_states_dict['v_delta_x']*0.3)
        agent_states_list.append(agent_states_dict['v_delta_z']*0.3)
        agent_states_list.append(agent_states_dict['player_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['player_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_me_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_me_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_basket_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_basket_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_ball_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_ball_r']*0.2)
        agent_states_list.append(agent_states_dict['facing_x'])
        agent_states_list.append(agent_states_dict['facing_y'])
        agent_states_list.append(agent_states_dict['facing_z'])
        agent_states_list.append(agent_states_dict['block_remain_best_time'])
        agent_states_list.append(agent_states_dict['block_remain_time'])
        agent_states_list.append(agent_states_dict['is_out_three_line'])
        agent_states_list.append(agent_states_dict['is_ball_owner'])
        agent_states_list.append(agent_states_dict['own_ball_duration']*0.2)
        agent_states_list.append(agent_states_dict['cast_duration'])
        agent_states_list.append(agent_states_dict['power']* 0.001)
        agent_states_list.append(agent_states_dict['is_cannot_dribble'])
        agent_states_list.append(agent_states_dict['is_pass_receiver'])
        agent_states_list.append(agent_states_dict['is_marking_opponent'])
        agent_states_list.append(agent_states_dict['is_team_own_ball'])
        agent_states_list.append(agent_states_dict['inside_defence'])
        is_my_team = onehot(int(agent_states_dict['is_my_team']), 2)
        agent_states_list += is_my_team
        player_state = onehot(int(agent_states_dict['player_state']), 6)
        agent_states_list += player_state
        skill_state = onehot(int(agent_states_dict['skill_state']), 27)
        agent_states_list += skill_state
        return np.array(agent_states_list)
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model().to('cpu')
    def sample_action(self, states):
        new_states = []
        for state in states:
            new_states.append(state[np.newaxis, :])
        new_states = [torch.tensor(state) for state in new_states]
        value, probs = self.model(new_states)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action.detach().numpy().item(),log_probs
    def get_actions(self, states_dic):
        actions = {}
        self.log_probs = {}
        for key in states_dic.keys():
            states = states_dic[key]
            actions[key], self.log_probs[key] = self.sample_action(states)
        return actions

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

class CustomedAgent():
    def __init__(self):
        self.model = None
        self.wrapper = SimpleWrapper({})
        self.policy = Policy()
        self.load()
    def load(self):
        curr_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(curr_path, 'bc_model')
        self.policy.load_model(model_path)

    def act(self, state_infos):
        states, infos = self.wrapper.state_wrapper(state_infos)
        actions = self.policy.get_actions(states)
        return actions

if __name__ == '__main__':
    import json
    agent = CustomedAgent()
    with open('states.json', 'r') as f:
        state_infos = json.load(f)
    print(agent.act(state_infos))