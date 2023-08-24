import numpy as np
from utils import onehot

class RLWrapper:
    def __init__(self, config):
        self.config = config
        self.observation_space = None
        self.action_space = None
    def states_wrapper(self, state_infos):
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
    
    def rewards_wrapper(self, state_infos):
        rewards = {}
        for key in state_infos.keys():
            infos = state_infos[key][0]
            state_event_reward = self.hanlde_state_event_rewards(infos)
            node_event_reward = self.handle_node_event_rewards(infos)
            dazhao_reward = self.handle_dazhao_rewards(infos)
            tot_reward = state_event_reward + node_event_reward + dazhao_reward
            rewards[key] = tot_reward
        return rewards
    
    def hanlde_state_event_rewards(self,infos):
        state_event_reward = 0
        if infos.get("state_event",None) is not None:
            state_event_dict = infos["state_event"]
            if state_event_dict.get("not_ball_clear",None) is not None:
                state_event_reward -= 0.0032
            if state_event_dict.get("ball_clear",None) is not None:
                state_event_reward -= 0.0016
            if state_event_dict.get("attack_time_out",None) is not None:
                state_event_reward -= 4
            if state_event_dict.get("got_defended",None) is not None:
                state_event_reward -=0.2
            if state_event_dict.get("out_of_defend",None) is not None:
                state_event_reward +=0.2
            if state_event_dict.get("long_pass",None) is not None:
                state_event_reward -=0.2
        return state_event_reward
    def handle_node_event_rewards(self,infos):
        node_event_reward = 0
        if infos.get("shoot",None) is not None:
            shoot_dict = infos["shoot"]
            expect_score = (shoot_dict["two"]* 2 + shoot_dict["three"] * 3) * shoot_dict["hit_percent"]
            if shoot_dict["open_shoot"]:
                expect_score *= 1.2 # encourage open shoot
            self_try_reward = expect_score * shoot_dict["me"]
            ally_try_reward = expect_score * shoot_dict["ally"] * 0.5
            enemy_try_reward = - expect_score * (shoot_dict["enemy"]*0.8 + shoot_dict["opponent"] *0.2)
            inference_reward = shoot_dict["inference_degree"]*0.5
            shoot_reward = self_try_reward + ally_try_reward + enemy_try_reward + inference_reward
            node_event_reward += shoot_reward
        if infos.get("steal",None) is not None:
            steal_dict = infos["steal"]
            hit_percent = steal_dict["hit_percent"]
            self_steal_reward = hit_percent * steal_dict["me"] * 0.5
            self_target_steal_reward = - hit_percent * steal_dict["target"] * 0.5
            steal_reward = self_steal_reward + self_target_steal_reward
            node_event_reward += steal_reward
        if infos.get("block",None) is not None:
            block_dict = infos["block"]
            expected_score = block_dict["expected_score"]
            hit_percent = block_dict["hit_percent"]
            self_block_reward = expected_score * hit_percent * block_dict["me"]
            ally_block_reward = expected_score * hit_percent * block_dict["ally"]*0.8
            self_target_block_reward = - expected_score * hit_percent * block_dict["target"]
            enemy_block_reward = - expected_score * hit_percent * block_dict["enemy"]*0.5
            block_reward = self_block_reward + ally_block_reward + self_target_block_reward + enemy_block_reward
            node_event_reward += block_reward
        if infos.get("rebound",None) is not None:
            rebound_dict = infos["rebound"]
            self_rebound_reward = rebound_dict["me"]*0.3
            enemy_rebound_reward = - rebound_dict["enemy"]*0.3
            rebound_reward = self_rebound_reward + enemy_rebound_reward
            node_event_reward += rebound_reward
        if infos.get("pickup",None) is not None:
            pickup_dict = infos["pickup"]
            self_pickup_reward = pickup_dict["me"]*pickup_dict["success"]*0.3
            enemy_pickup_reward = - pickup_dict["enemy"]*pickup_dict["success"]*0.3
            pickup_reward = self_pickup_reward + enemy_pickup_reward
            node_event_reward += pickup_reward
        if infos.get("screen",None) is not None:
            screen_dict = infos["screen"]
            screen_reward = 0
            if screen_dict["position"] <3:
                screen_reward += screen_dict["me"]*0.2
            node_event_reward += screen_reward
        return node_event_reward
    def handle_dazhao_rewards(self,infos):
        dazhao_reward = 0
        if len(infos["end_values"]) > 0:
            end_values_dict = infos["end_values"]
            dazhao_reward += end_values_dict["my_dazhao_cnt"]*0.2
        return dazhao_reward


class BCWrapper():
    ''' Simple Wrapper for Baseline
    '''
    def __init__(self, config):
        self.config = config

    def state_wrapper(self, states_dict):
        global_state = self.handle_global_states(states_dict['global_states']) # 
        self_state = self.handle_agent_states(states_dict['self_state'])
        ally0_state = self.handle_agent_states(states_dict['ally_0_state'])
        ally1_state = self.handle_agent_states(states_dict['ally_1_state'])
        enemy0_state = self.handle_agent_states(states_dict['enemy_0_state'])
        enemy1_state = self.handle_agent_states(states_dict['enemy_1_state'])
        enemy2_state = self.handle_agent_states(states_dict['enemy_2_state'])
        states = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state]
        return states

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
        agent_states_list.append(agent_states_dict['player_to_me_dis_y']*0.2) # z
        agent_states_list.append(agent_states_dict['basket_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_y']*0.2)# z
        agent_states_list.append(agent_states_dict['ball_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_y']*0.2)# z
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
    