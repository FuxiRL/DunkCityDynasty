import grpc
import psutil
import subprocess
from concurrent import futures

import DunkCityDynasty.env.multi_machine.machine_comm_pb2 as machine_comm_pb2
import DunkCityDynasty.env.multi_machine.machine_comm_pb2_grpc as machine_comm_pb2_grpc


CLIENT_PATH = r'path-to-game-client' # client path
GAME_SERVER_IP = '127.0.0.1' # game server ip
GAME_SERVER_PORT = 18000 # game server port
MACHINE_SERVER_PORT = 6667 # machine server port

CLIENT_PID_DICT = {}

class ClientCom(machine_comm_pb2_grpc.ClientCommServicer):
    def Cmd(self, request, context):
        """proto cmd
        """
        client_id = request.client_id
        cmd = request.cmd
        rl_server_ip = request.rl_server_ip
        rl_server_port = request.rl_server_port
        user_name = request.user_name

        if cmd == 'start_client':
            pid = self._start_client(rl_server_ip, rl_server_port, user_name)
            CLIENT_PID_DICT[client_id] = pid
            return machine_comm_pb2.Reply(msg='ok')
        
        elif cmd == 'close_client':
            if client_id in CLIENT_PID_DICT:
                self._close_client(CLIENT_PID_DICT[client_id])
                del CLIENT_PID_DICT[client_id]
                return machine_comm_pb2.Reply(msg='ok')
            else:
                return machine_comm_pb2.Reply(msg='no client')
            
        elif cmd == "check_client":
            client_pid = CLIENT_PID_DICT.get(client_id, -1)
            if client_id in CLIENT_PID_DICT and psutil.pid_exists(client_pid):
                return machine_comm_pb2.Reply(msg='ok')
            else:
                return machine_comm_pb2.Reply(msg='no client')

    def _start_client(self, rl_server_ip, rl_server_port, user_name):
        """start game client
        """
        # run game client 
        cmd = f'{CLIENT_PATH}/Lx33.exe {GAME_SERVER_IP} {GAME_SERVER_PORT} {rl_server_ip} {rl_server_port} {user_name}'
        p = subprocess.Popen(cmd, shell=False)
        return p.pid

    def _close_client(self, pid):
        """close game client
        """
        cmd = f"taskkill /F /PID {pid}"
        subprocess.call(cmd, shell=False)


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    machine_comm_pb2_grpc.add_ClientCommServicer_to_server(ClientCom(), server)
    server.add_insecure_port(f'[::]:{MACHINE_SERVER_PORT}')
    server.start()
    server.wait_for_termination()
