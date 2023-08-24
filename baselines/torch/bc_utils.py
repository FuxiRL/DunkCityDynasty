import jsonlines
import random
import numpy as np
from glob import glob

def get_file_names(dir_path):
    '''get all file names in the dir_path'''
    file_names = glob(dir_path + '/*/*.json')
    file_names = [file_name.replace('\\','/') for file_name in file_names]
    return file_names

def read_one_file(file_path):
    '''read one file and return the sa_traj'''
    with jsonlines.open(file_path) as reader:
        sa_traj = list(reader)
    return sa_traj

def convert_to_batch(sa_traj,wrapper):
    '''convert sa_traj to batch'''
    action_batch = []
    global_state_batch = []
    self_state_batch = []
    ally0_state_batch = []
    ally1_state_batch = []
    enemy0_state_batch = []
    enemy1_state_batch = []
    enemy2_state_batch = []
    for sa_dict in sa_traj:
        action = sa_dict['action']
        states_dict = sa_dict['state']
        states = wrapper.state_wrapper(states_dict)
        global_state_batch.append(states[0])
        self_state_batch.append(states[1])
        ally0_state_batch.append(states[2])
        ally1_state_batch.append(states[3])
        enemy0_state_batch.append(states[4])
        enemy1_state_batch.append(states[5])
        enemy2_state_batch.append(states[6])
        action_batch.append(action)
    global_state_batch = np.array(global_state_batch)
    self_state_batch = np.array(self_state_batch)
    ally0_state_batch = np.array(ally0_state_batch)
    ally1_state_batch = np.array(ally1_state_batch)
    enemy0_state_batch = np.array(enemy0_state_batch)
    enemy1_state_batch = np.array(enemy1_state_batch)
    enemy2_state_batch = np.array(enemy2_state_batch)
    states_batch = [global_state_batch, self_state_batch, ally0_state_batch, ally1_state_batch, enemy0_state_batch, enemy1_state_batch, enemy2_state_batch]
    action_batch = np.array(action_batch)
    return states_batch, action_batch

def sample_batch(file_pointers,wrapper):
    file_paths = random.sample(file_pointers, 3)
    actions = []
    global_state_batch = []
    self_state_batch = []
    ally0_state_batch = []
    ally1_state_batch = []
    enemy0_state_batch = []
    enemy1_state_batch = []
    enemy2_state_batch = []
    for file_path in file_paths:
        sa_traj = read_one_file(f"{file_path}")
        states_batch,action_batch = convert_to_batch(sa_traj,wrapper)
        global_state_batch.append(states_batch[0])
        self_state_batch.append(states_batch[1])
        ally0_state_batch.append(states_batch[2])
        ally1_state_batch.append(states_batch[3])
        enemy0_state_batch.append(states_batch[4])
        enemy1_state_batch.append(states_batch[5])
        enemy2_state_batch.append(states_batch[6])
        actions.append(action_batch)
    global_state_batch = np.concatenate(global_state_batch, axis=0)
    self_state_batch = np.concatenate(self_state_batch, axis=0)
    ally0_state_batch = np.concatenate(ally0_state_batch, axis=0)
    ally1_state_batch = np.concatenate(ally1_state_batch, axis=0)
    enemy0_state_batch = np.concatenate(enemy0_state_batch, axis=0)
    enemy1_state_batch = np.concatenate(enemy1_state_batch, axis=0)
    enemy2_state_batch = np.concatenate(enemy2_state_batch, axis=0)
    states = [global_state_batch, self_state_batch, ally0_state_batch, ally1_state_batch, enemy0_state_batch, enemy1_state_batch, enemy2_state_batch]
    actions = np.concatenate(actions, axis=0)
    return states, actions
    
