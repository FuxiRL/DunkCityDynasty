import os
import random
import jsonlines
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from baselines.common.wrappers import BCWrapper

def collate_fn(batch):
    global_state = rnn_utils.pad_sequence([torch.from_numpy(x[0]) for x in batch], batch_first=True, padding_value=0.0)
    self_state = rnn_utils.pad_sequence([torch.from_numpy(x[1]) for x in batch], batch_first=True, padding_value=0.0)
    ally0_state = rnn_utils.pad_sequence([torch.from_numpy(x[2]) for x in batch], batch_first=True, padding_value=0.0)
    ally1_state = rnn_utils.pad_sequence([torch.from_numpy(x[3]) for x in batch], batch_first=True, padding_value=0.0)
    enemy0_state = rnn_utils.pad_sequence([torch.from_numpy(x[4]) for x in batch], batch_first=True, padding_value=0.0)
    enemy1_state = rnn_utils.pad_sequence([torch.from_numpy(x[5]) for x in batch], batch_first=True, padding_value=0.0)
    enemy2_state = rnn_utils.pad_sequence([torch.from_numpy(x[6]) for x in batch], batch_first=True, padding_value=0.0)
    action = rnn_utils.pad_sequence([torch.from_numpy(x[7]) for x in batch], batch_first=True, padding_value=0.0)
    weight = rnn_utils.pad_sequence([torch.from_numpy(x[8]) for x in batch], batch_first=True, padding_value=0.0)
    return global_state, self_state,ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight

class BCDataset(Dataset):
    N_ACTION = 52
    MAX_LENGTH = 256
    def __init__(self,data_path,is_train=True):
        self.data_path = data_path
        self.is_train = is_train # True: train, False: test
        self.data_wrapper = BCWrapper(config=None)
        total_files, summit_level_dict = self.get_total_files()
        self.filed_files = self.file_filter(total_files, summit_level_dict)
        random.shuffle(self.filed_files)

    def __len__(self):
        return len(self.filed_files)
    
    def __getitem__(self, idx):
        states, actions = self.get_single_data(self.filed_files[idx])
        weights = self.get_data_weight(states, actions)
        global_state = np.array([state[0] for state in states])
        self_state = np.array([state[1] for state in states])
        ally0_state = np.array([state[2] for state in states])
        ally1_state = np.array([state[3] for state in states])
        enemy0_state = np.array([state[4] for state in states])
        enemy1_state = np.array([state[5] for state in states])
        enemy2_state = np.array([state[6] for state in states])
        action = np.array(actions)
        weight = np.array(weights)
        return global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight
    
    def sample(self, bz):
        test_indices = random.choices([i for i in range(len(self.filed_files))], k=bz)
        shuffle_state = []
        shuffle_action = []
        shuffle_weight = []
        for indice in test_indices:
            global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight = self.__getitem__(indice)
            global_state = torch.from_numpy(global_state)
            self_state = torch.from_numpy(self_state)
            ally0_state = torch.from_numpy(ally0_state)
            ally1_state = torch.from_numpy(ally1_state)
            enemy0_state = torch.from_numpy(enemy0_state)
            enemy1_state = torch.from_numpy(enemy1_state)
            enemy2_state = torch.from_numpy(enemy2_state)
            shuffle_state.append([global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state])
            shuffle_action.append(torch.from_numpy(action))
            shuffle_weight.append(torch.from_numpy(weight))
        return shuffle_state, shuffle_action, shuffle_weight

    def get_data_weight(self, states, actions):
        ''' Data Augmentation
        '''
        weights = [1. for _ in actions]
        for idx, (state, action) in enumerate(zip(states, actions)):
            character_id = state[1][0]
            # noop action
            if action == 0: weights[idx] *= 0.2
            # defend action
            if character_id != 82:
                if action == 16: weights[idx] *= 0.3
            else:
                if action == 17: weights[idx] *= 0.3
            # skill action
            if action >= 9: weights[idx] *= 5.
            # move action augmentation: move is different from last action
            if 0 < action < 9 and idx > 0 and action != actions[idx-1]: 
                pass
            else:
                weights[idx] *= 0.15
        return weights

    def get_single_data(self, filed_file):
        states = []
        actions = []
        with open(filed_file, 'r') as f:
            for line in jsonlines.Reader(f):
                state = self.data_wrapper.state_wrapper(line['state'])
                action = line['action']
                states.append(state)
                actions.append(action)
        return states, actions

    def get_total_files(self):
        total_files = []
        summit_level_dict = {}
        for folder_name in tqdm(os.listdir(self.data_path), desc='[dataset] get total files: '):
            # set DATA_RELEASE_9 as test set, others as train set
            if 'DATA_RELEASE' in folder_name:
                sector = int(folder_name.split('_')[-1])
                if self.is_train:
                    if sector == 9:
                        continue
                else:
                    if sector != 9:
                        continue
            # get total files
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.isdir(folder_path): # 过滤掉README.md
                for role_name in os.listdir(folder_path):
                    role_folder_path = os.path.join(folder_path, role_name)
                    if os.path.isdir(role_folder_path): # 过滤掉file_index.json
                        for file_name in os.listdir(role_folder_path):
                            file_path = os.path.join(role_folder_path, file_name)
                            total_files.append(file_path)
                
                file_index_path = os.path.join(folder_path, 'file_index.json')
                file_index_obj = json.load(open(file_index_path, 'r'))
                for file_index_obj_role_name in file_index_obj:
                    for filename in file_index_obj[file_index_obj_role_name]:
                        summit_level_dict[filename] = file_index_obj[file_index_obj_role_name][filename]['summit_level']
        return total_files, summit_level_dict

    def file_filter(self, total_files, summit_level_dict):
        # 指定summit level在【15，50】
        filed_files = []
        for file_path in tqdm(total_files, desc='[dataset] filte data: '):
            file_path = file_path.replace('\\', '/')
            file_name = file_path.split('/')[-1].split('.')[0].split('_')[1]
            summit_level = summit_level_dict[file_name]
            if summit_level >= 15 and summit_level <= 50 and os.path.getsize(file_path) > 1024:
                filed_files.append(file_path)
        
        return filed_files
