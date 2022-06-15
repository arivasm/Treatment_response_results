import torch
import pandas as pd
import pdb
import tqdm
import os 
import numpy as np
import argparse
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from warnings import filterwarnings
filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, 
                e_dict, 
                r_dict, 
                train_data, 
                already_ts_dict, 
                already_hs_dict,
                num_ng):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(train_data.values)
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict
        self.num_ng = num_ng
    
    def sampling(self, head, rel, tail):
        already_ts = torch.tensor(self.already_ts_dict[(head.item(), rel.item())])
        already_hs = torch.tensor(self.already_hs_dict[(tail.item(), rel.item())])
        neg_pool_t = torch.ones(len(self.e_dict))
        neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero()
        neg_pool_h = torch.ones(len(self.e_dict))
        neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero()
        neg_t = neg_pool_t[torch.randint(len(neg_pool_t), (self.num_ng//2,))]
        neg_h = neg_pool_h[torch.randint(len(neg_pool_h), (self.num_ng//2,))]
        return neg_t, neg_h
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        neg_t, neg_h = self.sampling(head, rel, tail)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng//2, -1), neg_t], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng//2, -1)], dim=1)
        sample = torch.cat([torch.tensor([head, rel, tail]).unsqueeze(0), neg_t, neg_h], dim=0)
        return sample

class TreatmentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = torch.tensor(data.values)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DrugTreatmentPU(torch.nn.Module):
    def __init__(self, emb_dim, terms_dict, iprs_dict, do, prior, pu, lmbda):
        super().__init__()
        self.emb_dim = emb_dim
        self.do = torch.nn.Dropout(do)
        self.prior = prior
        self.pu = pu
        self.fc_1 = torch.nn.Linear(1280, emb_dim)
        self.fc_2 = torch.nn.Linear(emb_dim, len(terms_dict))
        self.lmbda = lmbda

        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_2.weight.data)

    def pur_loss(self, pred, label):
        p_above = - (torch.nn.functional.logsigmoid(pred) * label).sum() / label.sum()
        p_below = - (torch.nn.functional.logsigmoid(-pred) * label).sum() / label.sum()
        u = - torch.nn.functional.logsigmoid((pred * label).sum() / label.sum() - (pred * (1 - label)).sum() / (1 - label).sum())
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above   

    def pu_loss(self, pred, label, lmbda):
        # pos_label = (label == 1).float()
        # unl_label = (label == 0).float() + (label == -1).float()

        # p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        # p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        # u = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        # if u > self.prior * p_below:
        #     return self.prior * p_above - self.prior * p_below + u
        # else:
        #     return self.prior * p_above


        pos_label = (label == 1).float()
        unl_label = (label == 0).float()
        neg_label = (label == -1).float()

        p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        u_0 = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        if neg_label.sum() > 0:
            u_1 = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * neg_label).sum() / neg_label.sum()
            u = (u_0 + lmbda * u_1) / (1 + lmbda)
        else:
            u = u_0
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above

        # # PU Learning
        # p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        # p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        # u = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        # if u > self.prior * p_below:
        #     pu_loss = self.prior * p_above - self.prior * p_below + u
        # else:
        #     pu_loss = self.prior * p_above

        # # PN Learning
        # if neg_label.sum() > 0:
        #     # pn_loss = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum() - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * neg_label).sum() / neg_label.sum()
        #     pn_loss = - torch.nn.functional.logsigmoid((pred * pos_label).sum() / pos_label.sum() - (pred * neg_label).sum() / neg_label.sum())
        # else:
        #     pn_loss = 0
        # return pu_loss + lmbda * pn_loss
    
    def pn_loss(self, pred, label):
        pos = - (torch.nn.functional.logsigmoid(pred) * label).sum() / label.sum()
        neg = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * (1 - label)).sum() / (1 - label).sum()
        return pos + neg

    def forward(self, X, Y):
        # mid = torch.nn.functional.leaky_relu(self.do(self.fc_1(X[:, :len(iprs_dict)])))
        # X_pred = self.fc_2(torch.cat([mid, X[:, len(iprs_dict):]], dim=-1))
        X_pred = self.fc_2(torch.nn.functional.leaky_relu(self.do(self.fc_1(X))))
        if self.pu == 0:
            return self.pn_loss(X_pred, Y)
        elif self.pu == 1:
            return self.pu_loss(X_pred, Y, self.lmbda)
        elif self.pu == 2:
            return self.pur_loss(X_pred, Y)
        
    def predict(self, X):
        # mid = torch.nn.functional.leaky_relu(self.do(self.fc_1(X[:, :len(iprs_dict)])))
        # return torch.sigmoid(self.fc_2(torch.cat([mid, X[:, len(iprs_dict):]], dim=-1)))
        return torch.sigmoid(self.fc_2(torch.nn.functional.leaky_relu(self.do(self.fc_1(X)))))


def read_data(cfg):
    train_data_kg = []
    train_data_tr = []
    with open(cfg.root + cfg.dataset + '/train_' + cfg.fold + '.ttl') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[1] == 'ex:belong_to':
                if line[2] == 'ex:effective':
                    train_data_tr.append([line[0], 1])
                elif line[2] == 'ex:low_effect':
                    train_data_tr.append([line[0], 0])
                else:
                    raise ValueError
            else:
                train_data_kg.append(line)
    train_data_kg = pd.DataFrame(train_data_kg, columns=['h', 'r', 't'])
    train_data_tr = pd.DataFrame(train_data_tr, columns=['tr', 'label'])
    
    test_data = []
    with open(cfg.root + cfg.dataset + '/test_' + cfg.fold + '.ttl') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[2] == 'ex:effective':
                test_data.append([line[0], 1])
            elif line[2] == 'ex:low_effect':
                test_data.append([line[0], 0])
            else:
                raise ValueError
    test_data = pd.DataFrame(test_data, columns=['tr', 'label'])
    
    all_e = set(train_data_kg['h'].unique()) | set(train_data_kg['t'].unique()) | set(train_data_tr['tr'].unique()) | set(test_data['tr'].unique())
    all_r = set(train_data_kg['r'].unique())
    
    e_dict = {k: v for v, k in enumerate(all_e)}
    r_dict = {k: v for v, k in enumerate(all_r)}
    
    train_data_kg.h = train_data_kg.h.map(e_dict)
    train_data_kg.r = train_data_kg.r.map(r_dict)
    train_data_kg.t = train_data_kg.t.map(e_dict)
    train_data_tr.tr = train_data_tr.tr.map(e_dict)
    test_data.tr = test_data.tr.map(e_dict)
    
    already_ts_dict = {}
    already_hs_dict = {}
    already_ts = train_data_kg.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs = train_data_kg.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts:
        already_ts_dict[(record[0], record[1])] = record[2]
    for record in already_hs:
        already_hs_dict[(record[0], record[1])] = record[2]
    
    return e_dict, r_dict, train_data_kg, train_data_tr, test_data, already_ts_dict, already_hs_dict

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data/', type=str)
    parser.add_argument('--dataset', default='G1', type=str)
    parser.add_argument('--fold', default='1', type=str)
    # Tunable
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--prior', default=0.0001, type=float)
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--pu', default=1, type=int)
    parser.add_argument('--num_ng', default=4, type=int)
    # parser.add_argument('--lmbda', default=1, type=float)
    # Untunable
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--valid_interval', default=20, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--tolerance', default=3, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    # save_root = f'../tmp/dataset_{cfg.dataset}_pu_{cfg.pu}_lmbda_{cfg.lmbda}_bs_{cfg.bs}_lr_{cfg.lr}_wd_{cfg.wd}_do_{cfg.do}_prior_{cfg.prior}_emb_dim_{cfg.emb_dim}/'
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)
    
    e_dict, r_dict, train_data_kg, train_data_tr, test_data, already_ts_dict, already_hs_dict = read_data(cfg)
    print(f'N Entities:{len(e_dict)}\nN Relations:{len(r_dict)}')
    train_dataset_kg = KnowledgeGraphDataset(e_dict, r_dict, train_data_kg, already_ts_dict, already_hs_dict, cfg.num_ng)
    train_dataset_tr = TreatmentDataset(train_data_tr)
    test_dataset = TreatmentDataset(test_data)
    
    train_dataloader_kg = torch.utils.data.DataLoader(dataset=train_dataset_kg, 
                                                    batch_size=cfg.bs, 
                                                    num_workers=cfg.num_workers, 
                                                    shuffle=True, 
                                                    drop_last=True)
    train_dataloader_tr = torch.utils.data.DataLoader(dataset=train_dataset_tr, 
                                                    batch_size=cfg.bs//4, 
                                                    num_workers=cfg.num_workers, 
                                                    shuffle=True, 
                                                    drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    batch_size=cfg.bs//4, 
                                                    num_workers=cfg.num_workers, 
                                                    shuffle=False, 
                                                    drop_last=False)                                 
    model = DrugTreatmentPU(cfg.emb_dim, e_dict, r_dict, cfg.do, cfg.prior, cfg.pu, cfg.lmbda)
    model = model.to(device)
    # tolerance = cfg.tolerance
    # max_fmax = 0
    # # min_loss = 100000
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:')
    #     model.train()
    #     avg_loss = []
        if cfg.verbose == 1:
            train_dataloader_kg = tqdm.tqdm(train_dataloader_kg)
            train_dataloader_tr = tqdm.tqdm(train_dataloader_tr)
        for batch in zip(train_dataloader_kg, train_dataloader_tr):
            batch_kg = batch[0].to(device)
            batch_tr = batch[1].to(device)
    #         loss = model(X, Y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         avg_loss.append(loss.item())
    #     avg_loss = round(sum(avg_loss)/len(avg_loss), 6)
    #     print(f'Loss: {avg_loss}')
    #     if (epoch + 1) % cfg.valid_interval == 0:
    #         model.eval()
    #         fmax = validate(model, valid_dataloader, device, cfg.verbose)
    #         if fmax > max_fmax:
    #             max_fmax = fmax
    #             tolerance = cfg.tolerance
    #         else:
    #             tolerance -= 1
    #         # if avg_loss < min_loss:
    #         #     min_loss = avg_loss
    #         #     tolerance = cfg.tolerance
    #         # else:
    #         #     tolerance -= 1
    #         torch.save(model.state_dict(), save_root + (str(epoch + 1)))
    #     if tolerance == 0:
    #         print(f'Best performance at epoch {epoch - cfg.tolerance * cfg.valid_interval + 1}')
    #         model.eval()
    #         model.load_state_dict(torch.load(save_root + str(epoch - cfg.tolerance * cfg.valid_interval + 1)))
    #         # model.load_state_dict(torch.load(save_root + '420'))
    #         test_df = test(model, test_dataloader, device, cfg.verbose, test_data, terms_dict, go)
    #         evaluate(cfg.root[:-1], cfg.dataset, test_df)
    #         break

