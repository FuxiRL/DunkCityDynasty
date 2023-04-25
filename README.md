![](./assets/Fuxi_logo.png)

# Dunk City Dynasty: Asynchronous Multi-Agent Basketball Environment

This repository contains an RL environment based on a commercial online basketball game named [Dunk City Dynasty](https://www.dunkcitymobile.com/). It was created by the [Netease Leihuo Technology](https://leihuo.163.com/) and the [Netease FuXi AI lab](https://fuxi.163.com/) for research purposes.

<img src="./assets/game_pic.jpg" style="zoom:67%;" />

## Content

- [Overview](#overview)
- [QuickStart](#quickStart)
- [QA](#QA)
- [Appendix](#appendix)

## Overview

Dunk City Dynasty is a brand new multiplayer online basketball game developed by NetEase Games. In the game, players can choose their favorite characters and participate in 3v3 street basketball matches. The player-controlled characters in the game are mostly star players from the National Basketball Association (NBA) in the United States, such as LeBron James, Kevin Durant, Stephen Curry, and so on, each with their own characteristics and exclusive skills. Players need to use their skills and strategies to win matches and improve their level and ranking. For more information about the game, please refer to the official website at https://www.dunkcitymobile.com/.

### Game Challenge

- **Real-time Environment**
- **Asynchronous Decision-Making**
- **Policy Generalization**

### Observation Space

Same as traditional multi-agent environments, the observation space of the Dunk City Dynasty Environment consists of `global_state`, `self_state`,` ally state`, `enemy state`, and `legal action`. The `ally state` and `enemy state` contain the state features of each teammate and opponent, respectively. Details can be seen in [here](./assets/state.md).

### Action Space

In Dunk City Dynasty Environment, each role have various executed actions while sharing the same `movement actions` (Up, Up Right, Right, Down Right, Down, Down Left, Left, Up Left), `Noop action`, `Cancel action` and `Pass Ball action`. In addition to shared actions, each character also has their own `unique basic skills` and `special skills`. Totally, he action space is a 52-dimensional discrete action space.. Specific action space can be found in [the related documentation](./assets/role_action.xlsx).

### Game Event

Dunk City Dynasty Environment event contains `shoot event`, `steal event`, `block event`, `pickup event`, `rebound event` and `screen event`. Details can be seen in [here](./assets/event.md).

## QuickStart

### Installation

Everyone can install the Dunk City Dynasty Env via Github. First, download the repo via

```sh
git clone xxx.git
```

Then, to run the Dunk City Dynasty client, wine and several components are needed. We can install wine via the `install.sh` or using `docker`.

```sh
# (Only for Ubuntu System) install components
./install_deps.sh

# install components via docker
docker build -t dunk_city_dynasty_env:v1.0 .
```

Finally, install the relevant python packages. (Currently, we only support python=3.8)

```sh
conda create -n dunk_city_dynasty python=3.8
conda activate dunk_city_dynasty

pip install -r requirements.txt
```

### Get start

After installing the corresponding components, we can run the Dunk City Dynasty Env with the following python code.

- `Alt+Enter`: game screen scaling.

```python
config = {
    'id': 1,
    'env_setting': 'win', # or 'linux' for ubuntu system
    'client_path': 'path-of-game-client',
    'rl_server_ip': '127.0.0.1', # ip of the rl server
    'rl_server_port': 6666,  # port of the rl server
    'game_server_ip': 'xxxxxxx', # ip of the game server (we will provide a public IP address and port later)
    'game_server_port': 6667, # port of the game server (we will provide a public IP address and port later)
    'machine_server_ip': '', # ip of remote machineserver (for multi machine setting)
    'machine_server_port': 0, # port of remote machineserver (for multi machine setting)
    'episode_horizon': 100000 # max game steps
    }

env = GymEnv(config)

states, infos = env.reset()
while True:
    action = {key: random.randint(0, 20) for key in states}
    states, rewards, dones, truncated, infos = env.step(action)
    print(action)
    if dones['__all__']:
        break
```

or, directly run the `get_start.py` file

```sh
# need to perform the relevant configuration in the get_start section first!!!
python get_start.py
```

### An easy training demo

```sh
python train_ppo.py
```

## QA

##### The game crashes when executing the `get_start.py` file.

If the game crashes when executing the get_start.py file for the first time, you can try executing it again.

##### When running get_start.py on an Ubuntu system, there is an issue with Wine getting stuck.

Try quitting get_start.py and running the get_start.py file again. 


## Appendix

- Client Path: 
- Human Data:

