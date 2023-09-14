import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def get_output_size_with_batch(model, input_size, dtype=torch.float):
    with torch.no_grad():
        output = model(torch.zeros([1, *input_size[1:]], dtype=dtype))
        output_size = [None] + list(output.size())[1:]
    return output_size

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
        self.global_state_len = 30 # lenth of global state
        self.linear_layer, self.linear_layer_out_dim = linear_layer([None, self.global_state_len], 64)
    def forward(self, x):
        x = x.float()
        x = self.linear_layer(x)
        return x

class AgentStateLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent_state_len = 73 # lenth of agent state
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
        if len(states) > 7: # action mask for rl training
            action_mask = states[7].float()
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
        if len(states) > 7:
            large_negative = torch.finfo(logits_p.dtype).min if logits_p.dtype == torch.float32 else 1e-9
            mask_logits_p = logits_p * action_mask + (1 - action_mask) * large_negative
            probs = nn.functional.softmax(mask_logits_p, dim=1)
            return value.float(), probs.float()
        else: # for bc training
            return logits_p.float()