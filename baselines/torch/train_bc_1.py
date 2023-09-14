import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
from utils import onehot
from bc_utils import get_file_names, sample_batch, read_one_file, convert_to_batch
from model import Model
from wrappers import BCWrapper
    
class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        return action.detach().numpy().item(),log_probs
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def evaluate(self, states, actions):
        value, probs = self.model(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze(dim=1))
        entropys = dist.entropy()
        return value, log_probs, entropys
    def sgd_iter(self, states, actions, returns, old_log_probs):
        batch_size = actions.shape[0]
        mini_batch_size = 32
        for _ in range(batch_size//mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield [states[i][rand_ids,:] for i in range(8)], actions[rand_ids,:], returns[rand_ids,:], old_log_probs[rand_ids,:]
    def update(self, states, actions):
        states = [torch.tensor(np.array(state),dtype=torch.float32) for state in states]
        weights = [0.1] + [0.2] * 8 + [1.2] * 43
        weights = torch.tensor(weights,dtype=torch.float32)
        actions = torch.tensor(actions,dtype=torch.long).unsqueeze(1)
        logit_p = self.model(states)
        predict_actions = logit_p.argmax(dim=1)
        acc = (predict_actions == actions.squeeze(dim=1)).sum().item() / len(actions)
        loss = F.cross_entropy(logit_p, actions.squeeze(dim=1), weight = weights)
        self.model.opt.zero_grad()
        loss.backward()
        self.model.opt.step()
        return loss.item(), acc
    
if __name__ == '__main__':
    
    human_data_dir = "./human_data"
    TOTAL_DIRS = [
        "DATA_RELEASE_NEW_HANDLED0",
        "DATA_RELEASE_NEW_HANDLED1",
        "DATA_RELEASE_NEW_HANDLED2",
    ]
    file_pointers = []
    for dir_name in TOTAL_DIRS:
        dir_name = f"{human_data_dir}/{dir_name}"
        file_names = get_file_names(dir_name)
        file_pointers += file_names
    tb_writer = SummaryWriter(f"./output/logs/")
    wrapper = BCWrapper({})
    policy = Policy()
    num_epochs = 100000
    for epoch in range(num_epochs):
        try: # avoid file read error
            states_batch, action_batch = sample_batch(file_pointers,wrapper)
            loss,acc = policy.update(states_batch,action_batch)
            if epoch % 10 == 0:
                tb_writer.add_scalar('loss', loss, epoch)
                tb_writer.add_scalar('acc', acc, epoch)
            if epoch % 2000 == 0:
                print(f"epoch:{epoch},loss:{loss},acc:{acc}")
                policy.save_model(f"./output/bc_model")
        except:
            pass