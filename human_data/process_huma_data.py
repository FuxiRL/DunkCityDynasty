import jsonlines

import numpy as np


def state_embedding(states):
    global_state = np.array(list(states['global_states'].values()))
    self_state = np.array(list(states['self_state'].values()))
    ally0_state = np.array(list(states['ally_0_state'].values()))
    ally1_state = np.array(list(states['ally_1_state'].values()))
    enemy0_state = np.array(list(states['enemy_0_state'].values()))
    enemy1_state = np.array(list(states['enemy_1_state'].values()))
    enemy2_state = np.array(list(states['enemy_2_state'].values()))

    observations = np.concatenate((global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state), axis=0)
    return list(observations)

def process_state(origin_file_path, target_file_path):
    human_data_processed = []
    with jsonlines.open(origin_file_path) as reader:
        data = list(reader)

    # convert data to ray offline data 
    for i in range(len(data)):
        each_data = {}
        each_data["type"] = "SampleBatch"
        state = state_embedding(data[i]['state']) 
        state = [state]
        each_data['obs'] = state

        if i == len(data)-1:
            each_data['new_obs'] = [state_embedding(data[i]['state'])]
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

    # save 
    with jsonlines.open(target_file_path, mode='w') as writer:
        writer.write_all(human_data_processed)



if __name__ == '__main__':
    ORIGIN_PATH = './human_data.json'
    TARGET_PATH = './human_data_processed.json'

    process_state(ORIGIN_PATH, TARGET_PATH)
