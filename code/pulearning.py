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
    def __init__(self, e_dict, r_dict, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.do = torch.nn.Dropout(cfg.do)
        self.prior = cfg.prior
        self.loss_type = cfg.loss_type
        self.base_model = cfg.base_model
        self.fc_1 = torch.nn.Linear(cfg.emb_dim, 1)
        # self.fc_2 = torch.nn.Linear(cfg.emb_dim//2, 1)
        self.e_embedding = torch.nn.Embedding(len(e_dict), cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(len(r_dict), cfg.emb_dim)

        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_2.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)

    def pu_loss(self, pred):
        p_above = - (torch.nn.functional.logsigmoid(pred[:, 0])).mean()
        p_below = (torch.log(1 - torch.sigmoid(pred[:, 0]) + 1e-10)).mean()
        u = - (torch.log(1 - torch.sigmoid(pred[:, 1:]) + 1e-10)).mean()
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above

    def pn_loss(self, pred):
        loss_pos = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        loss_neg = - torch.log(1 - torch.sigmoid(pred[:, 1:]) + 1e-10).mean()
        return (loss_pos + loss_neg) / 2

    def _DistMult(self, h_emb, r_emb, t_emb):
        return (h_emb * r_emb * t_emb).sum(dim=-1)
    
    def _TranE(self, h_emb, r_emb, t_emb):
        pass
    
    def _forward_kg(self, data):
        h_emb = self.e_embedding(data[:, :, 0])
        r_emb = self.r_embedding(data[:, :, 1])
        t_emb = self.e_embedding(data[:, :, 2])
        if self.base_model == 'DistMult':
            return self._DistMult(h_emb, r_emb, t_emb)
        else:
            raise ValueError

    def get_loss_kg(self, data):
        pred = self._forward_kg(data)
        if self.loss_type == 'pn':
            return self.pn_loss(pred)
        elif self.loss_type == 'pu':
            return self.pu_loss(pred)
        else:
            raise ValueError

    def _forward_tr(self, data):
        pos = torch.index_select(data, 0, (data[:, 1] == 1).nonzero().squeeze(-1))
        neg = torch.index_select(data, 0, (data[:, 1] == 0).nonzero().squeeze(-1))
        e_emb_pos = self.e_embedding(pos[:, 0])
        e_emb_neg = self.e_embedding(neg[:, 0])
        # pred_pos = self.fc_2(torch.nn.functional.leaky_relu(self.do(self.fc_1(e_emb_pos))))
        # pred_neg = self.fc_2(torch.nn.functional.leaky_relu(self.do(self.fc_1(e_emb_neg))))
        pred_pos = self.fc_1(e_emb_pos)
        pred_neg = self.fc_1(e_emb_neg)
        return pred_pos, pred_neg
    
    def get_loss_tr(self, data):
        pred_pos, pred_neg = self._forward_tr(data)
        loss_pos = - torch.nn.functional.logsigmoid(pred_pos).mean()
        loss_neg = - torch.log(1 - torch.sigmoid(pred_neg) + 1e-10).mean()
        return (loss_pos * len(pred_pos) + loss_neg * len(pred_neg)) / (len(pred_pos) + len(pred_neg))
    
    def predict(self, data):
        e_emb = self.e_embedding(data[:, 0])
        return torch.sigmoid(self.fc_1(e_emb))
        # return torch.sigmoid(self.fc_2(torch.nn.functional.leaky_relu(self.do(self.fc_1(e_emb)))))

def read_data(cfg):
    
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
                    train_data_kg.append(line)
            elif line[1] == 'ex:no_belong_to':
                if line[2] == 'ex:low_effect':
                    train_data_tr.append([line[0], 1])
                elif line[2] == 'ex:effective':
                    train_data_tr.append([line[0], 0])
                else:
                    train_data_kg.append(line)
            else:
                train_data_kg.append(line)
    
    for train in train_data_tr:
        for test in test_data:
            if train == test:
                train_data_tr.remove(train)    
    
    test_data = pd.DataFrame(test_data, columns=['tr', 'label'])
    train_data_kg = pd.DataFrame(train_data_kg, columns=['h', 'r', 't'])
    train_data_tr = pd.DataFrame(train_data_tr, columns=['tr', 'label'])
    
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

# def get_f1(labels, preds):
#     N = len(labels)
#     as_pos = (preds > 0.5).astype(int)
#     as_neg = (preds < 0.5).astype(int)
#     tp = (as_pos * labels).sum() 
#     fp = (as_pos * (1 - labels)).sum()
#     fn = (as_neg * labels).sum()
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return round(precision, 4), round(recall, 4), round(f1, 4)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data/', type=str)
    parser.add_argument('--dataset', default='G2', type=str)
    parser.add_argument('--fold', default='1', type=str)
    # Tunable
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0001, type=float)
    parser.add_argument('--do', default=0.5, type=float)
    parser.add_argument('--prior', default=0.001, type=float)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--loss_type', default='pu', type=str)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--lmbda', default=1, type=float)
    parser.add_argument('--base_model', default='DistMult', type=str)
    # Untunable
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--valid_interval', default=5, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--tolerance', default=20, type=int)
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
                                                    batch_size=cfg.bs, 
                                                    num_workers=cfg.num_workers, 
                                                    shuffle=True, 
                                                    drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    batch_size=cfg.bs//4, 
                                                    num_workers=cfg.num_workers, 
                                                    shuffle=False, 
                                                    drop_last=False)                                 
    model = DrugTreatmentPU(e_dict, r_dict, cfg)
    model = model.to(device)
    tolerance = cfg.tolerance
    max_value = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.verbose == 1:
        train_dataloader_kg = tqdm.tqdm(train_dataloader_kg)
        train_dataloader_tr = tqdm.tqdm(train_dataloader_tr)
        test_dataloader = tqdm.tqdm(test_dataloader)
    for epoch in range(cfg.max_epochs):
        # print(f'Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        for batch in zip(train_dataloader_kg, train_dataloader_tr):
            batch_kg = batch[0].to(device)
            batch_tr = batch[1].to(device)
            loss_kg = model.get_loss_kg(batch_kg)
            loss_tr = model.get_loss_tr(batch_tr)
            loss = (cfg.lmbda * loss_kg + loss_tr) / (cfg.lmbda + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        avg_loss = round(sum(avg_loss)/len(avg_loss), 6)
        # print(f'Loss: {avg_loss}')
        if (epoch + 1) % cfg.valid_interval == 0:
            labels = []
            preds = []
            with torch.no_grad():
                model.eval()
                for batch in test_dataloader:
                    labels.append(batch[:, 1])
                    batch = batch.to(device)
                    preds.append(model.predict(batch))
            labels, preds = torch.cat(labels).numpy(), torch.cat(preds).cpu().numpy().flatten()
            auc = round(roc_auc_score(labels, preds), 4)
            aupr = round(average_precision_score(labels, preds), 4)
            precision, recall, thresholds = precision_recall_curve(labels, preds)
            f1_scores = 2 * recall * precision / (recall + precision)
            fmax = round(np.max(f1_scores), 4)
            th = thresholds[np.argmax(f1_scores)]
            print(f'AUC:{auc}\tAUPR:{aupr}\tFmax:{fmax}\tTh:{th}')
            # precision, recall, f1 = get_f1(labels, preds)
            # print(f'AUC:{auc}\tAUPR:{aupr}\tPrecision:{precision}\tRecall:{recall}\tF1:{f1}')
            if (auc + aupr + fmax) > max_value:
                max_value = (auc + aupr + fmax)
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            if tolerance == 0:
                break

