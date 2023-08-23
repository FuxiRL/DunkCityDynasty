import numpy as np


class CustomedAgent():
    def __init__(self):
        pass

    def act(self, obs):
        # return {key: 3 for key in obs}
        return {key: np.random.randint(4, 8) for key in obs}
