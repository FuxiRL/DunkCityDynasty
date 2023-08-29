import os
import re
import time
import grpc
import threading
import subprocess
import socketserver

import DunkCityDynasty.env.multi_machine.machine_comm_pb2 as machine_comm_pb2
import DunkCityDynasty.env.multi_machine.machine_comm_pb2_grpc as machine_comm_pb2_grpc
from DunkCityDynasty.utils.tcp_server import ThreadedTCPRequestHandler, CustomedThreadingTCPServer
from DunkCityDynasty.utils.util_fn import sleep


class BaseEnv():
    def __init__(self, config):
        self.config = config        

        # env config
        self.id = config['id']
        self.env_setting = config['env_setting']
        assert self.env_setting in ['win', 'linux', 'multi_machine']

        # client config
        self.client_path = config['client_path']
        self.rl_server_ip = config['rl_server_ip']
        self.rl_server_port = config['rl_server_port']
        self.game_server_ip = config['game_server_ip']
        self.game_server_port = config['game_server_port']
        self.user_name = config['user_name']

        if self.env_setting == 'linux':
            if 'xvfb_display' not in config:
                self.xvfb_display = 5
            else:
                self.xvfb_display = config['xvfb_display']

        if self.env_setting =='multi_machine':
            self.machine_server_ip = config['machine_server_ip']
            self.machine_server_port = config['machine_server_port']

        # training config
        self.episode_horizon = config['episode_horizon']

        # env hyperparameter
        self.total_agent = 6 # total game agent
        self.game_pid = None # game client pid
        self.step_cnt = 0 # game step count
        self.ep_step_cnt = 0 # episode step count, different from step_cnt
        self.ep_cnt = 0 # episode count

        # detect if a virtual desktop is being used
        if self.env_setting == 'linux':
            self.use_xvfb = True 
            self._start_virtual_desktop()
            if os.environ.get('DISPLAY'):
                self.use_xvfb = False

    def reset(self):
        """reset func
        """
        # set stream data & start tcp server
        self.stream_data = {
            i: {
                'state': None,
                'action': 0. # set default action
            }
            for i in range(self.total_agent)
        }
        self.stream_data['done'] = False
        self._start_tcp_server()

        # start game client
        self.game_pid = self._start_client()
        # get state        
        states = self._wait_for_state(min_player=3)

        # If the game does not respond for a long time, restart the game
        while states is None:
            self._close_client()
            time.sleep(1)
            self.game_pid = self._start_client()
            states = self._wait_for_state(min_player=3)

        # hyperparameter reset
        self.step_cnt = 0
        self.last_step_time = time.time()
        return states

    def step(self, action_dict):
        """step func
        """
        # set minimum game execution time (5ms)
        curr_time = time.time()
        delta_time = curr_time - self.last_step_time
        if delta_time < 0.005:
            sleep(0.005 - delta_time)

        # 1. set action
        self._set_action(action_dict)
        # 2. get new state from client 
        states = self._wait_for_state(min_player=3)
        if states is None:
            raise Exception("Game client did not respond")

        # 3. get game done info
        truncated, done = self._get_done(states)

        self.step_cnt += 1
        self.ep_step_cnt += 1
        self.last_step_time = time.time()
        return states, truncated, done
    
    def _get_agent_truncated(self, state_infos):
        """get agent truncated info
        """
        one_agent_truncated_flag = False
        agent_truncated = {}
        for key in state_infos.keys():
            infos = state_infos[key][0]
            if infos.get('end_values', None) is not None:
                if len(infos['end_values']) > 0:
                    one_agent_truncated_flag = True
                    agent_truncated[key] = True
                else:
                    agent_truncated[key] = False
            else:
                agent_truncated[key] = False
        return one_agent_truncated_flag, agent_truncated
    
    def _get_done(self, states):
        """get game done info
        """
        done = False
        truncated = {"__all__": False}
        one_agent_truncated_flag, agent_truncated = self._get_agent_truncated(states)
        truncated.update(agent_truncated)
        if one_agent_truncated_flag and self.ep_step_cnt >= 5: # avoid send end_values many times
            self.ep_step_cnt = 0
            self.ep_cnt += 1
            truncated["__all__"] = True

        # done via step cnt
        if self.step_cnt >= self.episode_horizon:
            done = True
            # close game client
            self._close_client()

            # stop tcp server
            self.stream_data['done'] = True
            self._close_tcp_server()

            return truncated, done
        
        # done via game time
        for key in states:
            if states[key][1]['global_state']['match_remain_time'] < 0.2:
                truncated["__all__"] = True
                done = True
                # close game client
                self._close_client()

                # stop tcp server
                self.stream_data['done'] = True
                self._close_tcp_server()

                break

        return truncated, done

    # ===================================================
    #   Game Client Script
    # ===================================================
    def _start_client(self):
        """start game client
        """
        if self.env_setting == 'win':
            # run game client 
            cmd = f"{self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name}"
            p = subprocess.Popen(cmd, shell=False)
            pids = [p.pid]

        elif self.env_setting == 'linux':
            # run game client
            if self.use_xvfb:
                cmd = f'export DISPLAY=:{self.xvfb_display} && wine {self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name}&'
            else:
                cmd = f'wine {self.client_path}/Lx33.exe {self.game_server_ip} {self.game_server_port} {self.rl_server_ip} {self.rl_server_port} {self.user_name}&'
            os.system(cmd)

            # get game client pid
            ps_lines = os.popen('ps -ef |grep Lx33.exe').readlines()
            pids = []
            for ps_line in ps_lines:
                if f'{self.rl_server_ip} {self.rl_server_port}' in ps_line:
                    temp = re.sub(' +', ' ', ps_line).split(' ')
                    if len(temp) > 1:
                        pids.append(int(temp[1]))

        elif self.env_setting == 'multi_machine':
            with grpc.insecure_channel(f'{self.machine_server_ip}:{self.machine_server_port}') as channel:
                stub = machine_comm_pb2_grpc.ClientCommStub(channel)
                resp = stub.Cmd(machine_comm_pb2.ClientCmd(
                    client_id=self.id,
                    cmd='start_client',
                    rl_server_ip=self.rl_server_ip,
                    rl_server_port=self.rl_server_port,
                    user_name=self.user_name,
                ))
            if resp.msg == 'ok':
                pids = [-1]
            else:
                raise Exception('error!!')
            
        return pids

    def _close_client(self):
        """close game client
        """
        if self.env_setting == 'win':
            for pid in self.game_pid:
                cmd = f"taskkill /F /PID {pid}"
                subprocess.call(cmd, shell=False)
        
        elif self.env_setting == 'linux':
            for pid in self.game_pid:
                cmd = f"kill -9 {pid}"
                os.system(cmd)
        
        elif self.env_setting == 'multi_machine':
            with grpc.insecure_channel(f'{self.machine_server_ip}:{self.machine_server_port}') as channel:
                stub = machine_comm_pb2_grpc.ClientCommStub(channel)
                resp = stub.Cmd(machine_comm_pb2.ClientCmd(
                    client_id=self.id,
                    cmd='close_client',
                    rl_server_ip=self.rl_server_ip,
                    rl_server_port=self.rl_server_port,
                ))
            if resp.msg != 'ok':
                raise Exception('error!!')

    def _start_virtual_desktop(self):
        """start virtual desktop
        """
        if self.env_setting != 'linux':
            return 
        
        cmd = f"Xvfb :{self.xvfb_display} -screen 0 300x300x24 -fbdir /var/tmp &"
        os.system(cmd)

    # ===================================================
    #  Tcp Server
    # ===================================================
    def _start_tcp_server(self):
        """start tcp server
        """
        # set tcp server
        socketserver.TCPServer.allow_reuse_address = True
        self.tcp_server = CustomedThreadingTCPServer((self.rl_server_ip, self.rl_server_port), ThreadedTCPRequestHandler)
        self.tcp_server.customed_set_data(self.stream_data)

        # start server thread
        self.server_thread = threading.Thread(target=self.tcp_server.serve_forever)
        self.server_thread.start()

    def _close_tcp_server(self):
        """close tcp server
        """
        self.tcp_server.shutdown()
        self.tcp_server.server_close()

    def _wait_for_state(self, min_player=0, max_wait_count=60):
        """wait the fixed time to accept env state
        """
        states = {}
        wait_count = 0
        while True:
            for i in range(self.total_agent):
                if self.stream_data[i]['state']:
                    states[i] = self.stream_data[i]['state']
                    self.stream_data[i]['state'] = None
            if len(states) > min_player:
                return states

            # sleep
            sleep(0.001)

            # timeout detection.
            wait_count += 0.001
            if wait_count > max_wait_count:
                return None

    def _set_action(self, action_dict):
        """set action to tcp server
        """
        for action_key in action_dict:
            self.stream_data[action_key]['action'] = int(action_dict[action_key])