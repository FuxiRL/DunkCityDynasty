import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 
from model import Model
from wrappers import RLWrapper
from DunkCityDynasty.env.gym_env import GymEnv

class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory = Memory()
        self.model = Model().to('cpu')
        self.update_step = 0

    def sample_action(self, states):
        new_states = []
        for state in states:
            new_states.append(state[np.newaxis, :])
        new_states = [torch.tensor(state) for state in new_states]
        value, probs = self.model(new_states)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action.detach().numpy().item(), log_probs
    def get_random_actions(self, states):
        action_masks = {key: np.argwhere(states[key][-1] == 1.).flatten() for key in states}
        return {key: random.choices(action_masks[key].tolist(), k=1)[0] for key in action_masks}
    def get_actions(self, states_dic):
        actions = {}
        self.log_probs = {}
        for key in states_dic.keys():
            states = states_dic[key]
            actions[key], self.log_probs[key] = self.sample_action(states)
        return actions
    def _compute_returns(self, rewards,dones,gamma=0.99):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(np.array(returns)).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
    def save_model(self, path, step):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/{step}")

class Exp:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

class Memory():
    def __init__(self) -> None:
        self.buffer = deque(maxlen=10000)
    def push(self,exps):
        self.buffer.append(exps)
    def handle(self, exps):
        states = [[] for _ in range(8)]
        for exp in exps:
            for i in range(8):
                states[i].append(exp.state[i])
        for i in range(8):
            states[i] = np.array(states[i])
        next_states = [[] for _ in range(8)]
        for exp in exps:
            for i in range(8):
                next_states[i].append(exp.next_state[i])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        dones = np.array([exp.truncated for exp in exps])
        old_log_probs =  [exp.log_probs for exp in exps]
        return states, next_states, actions, rewards, dones, old_log_probs
    def sample(self):
        if len(self.buffer) == 0: return None
        exps = self.buffer.popleft()
        return self.handle(exps) 
    
class StatsRecorder:
    def __init__(self) -> None:
        self.writters = {}
        self.writter_types = ['interact','policy']
        for writter_type in self.writter_types: 
            self.writters[writter_type] = SummaryWriter(f'./output/logs/{writter_type}')
    def add_scalar(self, tag, scalar_value, global_step):
        for writter_type in self.writter_types: 
            self.writters['interact'].add_scalar(tag=tag, scalar_value=scalar_value, global_step = global_step)
    def add_rewards(self, rewards, ep_cnt):
        for key in rewards.keys():
            self.writters['interact'].add_scalar(tag=f'rewards/member_{key}', scalar_value=rewards[key], global_step= ep_cnt)
    def add_policy_loss(self, loss, ep_cnt):
        self.writters['policy'].add_scalar(tag=f'loss/policy_loss', scalar_value = loss, global_step = ep_cnt)
    def add_value_loss(self, loss, ep_cnt):
        self.writters['policy'].add_scalar(tag=f'loss/value_loss', scalar_value = loss, global_step = ep_cnt)

def create_env(id=1):
    config = {
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
        'episode_horizon': 100000,
        'render': True,
    }
    wrapper = RLWrapper({})
    env = GymEnv(config, wrapper=wrapper)
    return env

def train(env, policy,stats_recorder=None):
    ep_cnt = 0
    for i in range(1000000):
        exps = []
        states, infos = env.reset()
        ep_rewards = defaultdict(int)
        ep_step = 0
        while True:
            actions = policy.get_actions(states)
            if ep_cnt < 50:
                actions = policy.get_random_actions(states)
            next_states , rewards, dones, truncated, infos = env.step(actions)
            ep_step += 1
            for key in rewards.keys():
                ep_rewards[key] += rewards[key]
            if truncated['__all__'] or ep_step >= 120:
                ep_step = 0
                ep_cnt += 1
                stats_recorder.add_rewards(ep_rewards, ep_cnt)
            share_keys = list(set(states.keys()) & set(next_states.keys()) & set(actions.keys()) & set(rewards.keys()) & set(dones.keys()))
            for key in share_keys:
                state = states[key]
                next_state = next_states[key]
                action = actions[key]
                reward = rewards[key]
                done = dones[key]
                truncat = truncated[key]
                log_probs = policy.log_probs[key]
                exp = Exp(state=state,next_state=next_state,action=action,reward=reward,done=done,log_probs=log_probs,truncated=truncat)
                exps.append(exp)
            if len(exps) >= 512:
                policy.memory.push(exps)
                policy.update(stats_recorder = stats_recorder)
                exps = []
            if dones['__all__']:
                break

            states = next_states

def test(env,policy):
    states, infos = env.reset()
    while True:
        actions = policy.get_actions(states)
        print(actions)
        next_states , rewards, dones, truncated, infos = env.step(actions)
        if dones['__all__']:
            break
        states = next_states

if __name__ == '__main__':
    env = create_env()
    policy = Policy()
    stats_recorder = StatsRecorder()
    train(env, policy, stats_recorder=stats_recorder)
    # test(env, policy)