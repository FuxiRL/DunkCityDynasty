import time
import threading
import socketserver

from DunkCityDynasty.utils.rlsdk import *
from DunkCityDynasty.utils.util_fn import sleep


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """TCP Service Handler for Dunk City Dynasty game client 

    [V1.0] Multi-thread async TCP handler

    Args:
        request: default setting
        client_address: default setting
        server: default setting
        data: stream data for state & action of reinforcement learning

    Attributes:
        [handle]: data transmission function
    """

    def __init__(self, request, client_address, server, data) -> None:
        self.MAX_TIME_INTERVAL = 0.05 # Maximum Allowable Downtime (50ms)
        self.stream_data = data
        self.last_take_action = {key: data[key]['action'] for key in data if isinstance(key, int)}
        super().__init__(request, client_address, server)

    def setup(self):
        """server setup
        """
        self.request.setsockopt(socketserver.socket.IPPROTO_TCP, socketserver.socket.TCP_NODELAY, True)

    def handle(self):
        """data transmission function
        """
        try:
            self.data = bytes()
            while True:
                self.data += self.request.recv(10240000)
                while True:
                    head_length, msg = unpack_photon_rpc_head(self.data)
                    if head_length == 0:
                        break
                    self.data = self.data[head_length:]

                    msg_type, result = RlsdkDeserializer.deserialize(msg)
                    if msg_type == RLSDKMsgType.STARTINFO:
                        response = RlsdkDeserializer.serialize_recv_start_info(result.transaction_id, True, '\"\"')
                        response_with_head = pack_photon_h(len(response)) + response
                        sleep(0.01)
                        self.request.sendall(response_with_head)
    
                    elif msg_type == RLSDKMsgType.STATES:
                        team_id, member_id = result.agent_id.team_id, result.agent_id.member_id
                        stream_idx = team_id * 3 + member_id
  
                        if result.state and self.stream_data[stream_idx]['state'] is None:
                            self.stream_data[stream_idx]['state'] = result.state

                            start_time = time.time()
                            while self.stream_data[stream_idx]['action'] is None: 
                                sleep(0.0001)
                                if self.stream_data['done']:
                                    return
                                # Exceeding the maximum allowable execution time interval!!!
                                if time.time() - start_time > self.MAX_TIME_INTERVAL:
                                    self.stream_data[stream_idx]['action'] = self.last_take_action[stream_idx]
                                    break
                    
                            response = RlsdkDeserializer.serialize_action(result.transaction_id, [self.stream_data[stream_idx]['action']])
                            response_with_head = pack_photon_h(len(response)) + response
                            self.request.sendall(response_with_head)

                            self.last_take_action[stream_idx] = self.stream_data[stream_idx]['action']
                            self.stream_data[stream_idx]['action'] = None

        except:
            pass
        finally:
            print("stop tcp server: " + threading.currentThread().name)


class CustomedThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCP Service for  Dunk City Dynasty game client 

    [V1.0] Shared memory version

    Attributes:
        [customed_set_data]: set shared memory
        [finish_request]: override default finish_request func with shared memory and thread lock
    """
    def customed_set_data(self, data):
        """set shared memory
        """
        self.stream_data = data

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass
        """
        self.rhc = self.RequestHandlerClass(request, client_address, self, self.stream_data)
        
    def server_close(self):
        super().server_close()
        for thread in self._threads.pop_all():
            thread.join(timeout=2) # avoid deadlock