import gymnasium as gym
import numpy as np

class RayBCWrapper():
    def __init__(self, config):
        self.config = config
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(174,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(52)

    def reset(self):
        pass

    def state_wrapper(self, states):
        # return states
        new_states = {}
        for key in states.keys():

            each_states = states[key][1]
            global_state = np.array(list(each_states['global_state'].values()))
            self_state = np.array(list(each_states['self_state'].values()))
            ally0_state = np.array(list(each_states['ally_0_state'].values()))
            ally1_state = np.array(list(each_states['ally_1_state'].values()))
            enemy0_state = np.array(list(each_states['enemy_0_state'].values()))
            enemy1_state = np.array(list(each_states['enemy_1_state'].values()))
            enemy2_state = np.array(list(each_states['enemy_2_state'].values()))
           
            action_mask = np.array(states[key][-1])
            observations = np.concatenate((global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask), axis=0)
            new_states[key] = observations

        return new_states

    def calc_each_reward(self, state):
        reward = 0
        event = state[0]
        event_keys = list(event.keys())
        if len(event_keys) == 0:
            return reward
        if 'shoot' in event_keys:
            shoot_event = event['shoot']
            score_avg = shoot_event['shoot_type_two'] * 2 + shoot_event['shoot_type_three'] * 3
            score_avg *= (shoot_event['goal_in']+2)
            assist_reward = 0.5* shoot_event['assist'] * score_avg
            reward += assist_reward
            shoot_reward = (1 * shoot_event['me_shoot'] + 0.5 * shoot_event['ally_shoot'] - 1 * shoot_event['enemy_shoot'] - 0.5* shoot_event['opposing_player_shoot']) * score_avg
            reward += shoot_reward
        if 'steal' in event_keys:
            steal_event = event['steal']
            steal_reward = 0.2* steal_event['steal'] + 1* steal_event['steal_success'] - 0.5* steal_event['stolen'] - 1* steal_event['stolen_success']
            reward += steal_reward
        if 'block' in event_keys:
            block_event = event['block']
            block_reward = 0.5* block_event['block'] + 1* block_event['block_success'] - 0.5* block_event['blocked'] - 1* block_event['blocked_success']
            reward += block_reward
        if 'rebound' in event_keys:
            rebound_event = event['rebound']
            rebound_reward = 0.2* rebound_event['rebound'] + 0.5* rebound_event['rebound_success']
            reward += rebound_reward
        if 'pickup' in event_keys:
            pickup_event = event['pickup']
            pickup_reward = 1* pickup_event['pickup'] + 2* pickup_event['pickup_success']
            reward += pickup_reward

        global_state = state[1]['global_state']
        ball_clear = global_state['ball_clear'] # 球是否出三分
        self_state = state[1]['self_state']
        is_team_own_ball = self_state['is_team_own_ball'] # 球权
        attack_remain_time = global_state['attack_remain_time'] # 进攻剩余时间
        reward -= (1-ball_clear) * 0.5
        reward -= (180 - attack_remain_time) * 0.01 * is_team_own_ball
        
        return reward

    def reward_wrapper(self, states):
        rewards = {}
        for key in states.keys():
            state = states[key]
            reward = self.calc_each_reward(state)
            rewards[key] = reward

        return rewards

    def info_wrapper(self, states):
        infos = {}
        for key in states.keys():
            infos[key] = {}
        return infos