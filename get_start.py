import time
import random
import numpy as np

from DunkCityDynasty.env.gym_env import GymEnv


class RandomAgent():
    def __init__(self):
        pass

    def act(self, states):
        action_masks = {key: np.argwhere(states[key][-52:] == 1.).flatten() for key in states}
        return {key: random.choices(action_masks[key].tolist(), k=1)[0] for key in action_masks}


def main():
    # env config
    # --- win env
    config = {
        'id': 1,
        'env_setting': 'win',
        'client_path': 'path-to-game-client',
        'rl_server_ip': '127.0.0.1',
        'rl_server_port': 42636,
        'game_server_ip': '127.0.0.1',
        'game_server_port': 18000,
        'machine_server_ip': '',
        'machine_server_port': 0,
        'episode_horizon': 100000
    }

    # --- linux env
    # config = {
    #     'id': 1,
    #     'env_setting': 'linux',
    #     'client_path': 'path-to-game-client',
    #     'rl_server_ip': '127.0.0.1',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '',
    #     'machine_server_port': 0,
    #     'episode_horizon': 100000
    # }

    # # --- multi_machine
    # config = {
    #     'id': 1,
    #     'env_setting': 'multi_machine',
    #     'client_path': '',
    #     'rl_server_ip': '10.219.204.81',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '10.219.204.76',
    #     'machine_server_port': 6667,
    #     'episode_horizon': 100000
    # }

    env = GymEnv(config)
    agent = RandomAgent()

    states, infos = env.reset()
    while True:
        action = agent.act(states)
        states, rewards, dones, truncated, infos = env.step(action)
        print(action)
        if dones['__all__']:
            break

if __name__ == '__main__':
    main()