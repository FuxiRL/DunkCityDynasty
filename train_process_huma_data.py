#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 15:48:34
LastEditor: JiangJi
LastEditTime: 2023-04-27 16:12:18
Discription: 
'''
import os
import jsonlines




def process_states(states):
    import numpy as np
    global_state = np.array(list(states['global_states'].values())) # 15
    self_state = np.array(list(states['self_state'].values())) # 29
    ally0_state = np.array(list(states['ally_0_state'].values()))
    ally1_state = np.array(list(states['ally_1_state'].values()))
    enemy0_state = np.array(list(states['enemy_0_state'].values())) # 26
    enemy1_state = np.array(list(states['enemy_1_state'].values()))
    enemy2_state = np.array(list(states['enemy_2_state'].values()))
    observations = np.concatenate((global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state), axis=0)
    return list(observations)
human_data_processed = []
with jsonlines.open(f"{os.getcwd()}\\human_data\\human_data.json") as reader:
    data = list(reader)
for j in range(6):
    
    for i in range(len(data)):
        each_data = {}
        each_data["type"] = "SampleBatch"
        state =  process_states(data[i]['state']) 
        state = [state]
        each_data['obs'] = state
        if i == len(data)-1:
            each_data['new_obs'] = [process_states(data[i]['state'])]
        else:
            each_data['new_obs'] = state
        each_data["action_prob"] = [0.1]
        each_data['actions'] = [data[i]['action']]
        each_data['rewards'] = [0]
        each_data['dones'] = [False]
        each_data["infos"] = [{}]
        each_data["eps_id"] = [0]
        each_data["agent_index"] = [0]
        each_data["unroll_id"] = [i]
        each_data["t"] = [i]
        each_data["weights"] = [1.0]
        human_data_processed.append(each_data)

with jsonlines.open('human_data_processed.json', mode='w') as writer:
    writer.write_all(human_data_processed)