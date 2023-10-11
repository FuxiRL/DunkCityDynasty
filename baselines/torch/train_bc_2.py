import sys,os;sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter 
from baselines.common.utils import all_seed
from bc_collector import BCDataset, collate_fn
from bc_utils import download_human_data
from baselines.common.model import Model

class Config:
    def __init__(self) -> None:
        self.seed = 0
        self.n_epoch = 10
        self.data_path = "./human_data"
        self.train_batchsize = 1
        self.test_batchsize = 4
        self.dataloader_n_workers = 4
        self.lr = 1e-4
        self.device = torch.device('cpu')

def train(cfg):
    # download_human_data(cfg.data_path)
    all_seed(cfg.seed)
    train_dataset = BCDataset(cfg.data_path,is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batchsize, shuffle=True, num_workers=cfg.dataloader_n_workers,collate_fn=collate_fn,)
    test_dataset = BCDataset(cfg.data_path,is_train=False)
    model = Model().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    tb_writer = SummaryWriter(f"./output/logs/")
    step = 0
    for i_epoch in range(cfg.n_epoch):
        for i_batch, (global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight) in enumerate(train_dataloader):
            # squeeze when batch_size = 1
            global_state = global_state.squeeze(0).to(cfg.device)
            self_state = self_state.squeeze(0).to(cfg.device)
            ally0_state = ally0_state.squeeze(0).to(cfg.device)
            ally1_state = ally1_state.squeeze(0).to(cfg.device)
            enemy0_state = enemy0_state.squeeze(0).to(cfg.device)
            enemy1_state = enemy1_state.squeeze(0).to(cfg.device)
            enemy2_state = enemy2_state.squeeze(0).to(cfg.device)
            action = action.squeeze(0).to(cfg.device)
            weight = weight.squeeze(0).to(cfg.device)
            # Forward
            _, logits = model([global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state])
            m = Categorical(logits=logits)
            loss = -m.log_prob(action) * weight
            loss = loss.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i_batch % 500 == 0:
                test_states, test_actions, _ = test_dataset.sample(cfg.test_batchsize)
                test_accuracy = []
                for test_state, test_action in zip(test_states, test_actions):
                    if len(test_action) == 0:
                        continue 
                    with torch.no_grad():
                        test_state = [inner_test_state.to(cfg.device) for inner_test_state in test_state]
                        test_action = test_action.to(cfg.device)
                        _, test_pred_logits = model(test_state)
                        test_pred_action = test_pred_logits.argmax(-1)
                        test_accuracy.append((test_pred_action == test_action).float().mean().detach().cpu().numpy())
                test_accuracy = sum(test_accuracy) / len(test_accuracy)
                step += 1
                print(f'Epoch: {i_epoch}, Batch: {i_batch}, Loss: {loss.item()}, test_acc: {test_accuracy}')
                tb_writer.add_scalar('loss', loss, step)
                tb_writer.add_scalar('acc', test_accuracy, step)
        torch.save(model.state_dict(), f"./output/bc_model")

if __name__ == '__main__':
    cfg = Config()
    train(cfg)