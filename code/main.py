import torch
import pandas as pd
import pdb
import tqdm
import os
import numpy as np
import argparse
import random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

from numpy import fft

from warnings import filterwarnings

filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        neg_t = neg_pool_t[torch.randint(len(neg_pool_t), (self.num_ng // 2,))]
        neg_h = neg_pool_h[torch.randint(len(neg_pool_h), (self.num_ng // 2,))]
        return neg_t, neg_h

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        neg_t, neg_h = self.sampling(head, rel, tail)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng // 2, -1), neg_t], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng // 2, -1)], dim=1)
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
        self.scoring_fct_norm = cfg.scoring_fct_norm
        if cfg.base_model == 'ConvKB':
            #  === ConvKB ===
            self.hidden_size = cfg.emb_dim
            self.kernel_size = cfg.kernel_size
            self.out_channels = cfg.out_channels
            self.conv1_bn = torch.nn.BatchNorm2d(1)
            self.conv_layer = torch.nn.Conv2d(1, self.out_channels, (self.kernel_size, 3))  # kernel size x 3
            self.conv2_bn = torch.nn.BatchNorm2d(self.out_channels)
            self.dropout = torch.nn.Dropout(cfg.convkb_drop_prob)
            self.non_linearity = torch.nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = torch.nn.Linear((self.hidden_size - self.kernel_size + 1) * self.out_channels, 1, bias=False)
        elif cfg.base_model == 'TransH':
            #  === TransH ===
            self.norm_vector = torch.nn.Embedding(len(r_dict), cfg.emb_dim)
        elif cfg.base_model == 'RotatE':
            # === RotatE ===
            # self.margin = cfg.margin
            self.margin = torch.nn.Parameter(torch.Tensor([cfg.margin]), requires_grad=False)
            self.epsilon = cfg.epsilon
            self.e_embedding = torch.nn.Embedding(len(e_dict), self.emb_dim*2)
            self.r_embedding = torch.nn.Embedding(len(r_dict), self.emb_dim)
            self.rel_embedding_range = torch.nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]),
                requires_grad=False
            )
            self.pi_const = torch.nn.Parameter(torch.Tensor([3.14159265358979323846]))
            self.fc_1 = torch.nn.Linear(self.emb_dim*2, 1)

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
        # print('pred.shape:', pred.shape)
        loss_pos = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        loss_neg = - torch.log(1 - torch.sigmoid(pred[:, 1:]) + 1e-10).mean()
        return (loss_pos + loss_neg) / 2

    def _DistMult(self, h_emb, r_emb, t_emb):
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def _TransE(self, h_emb, r_emb, t_emb):
        return - torch.norm(h_emb + r_emb - t_emb, p=self.scoring_fct_norm, dim=-1)

    def _TransH(self, h_emb, r_emb, t_emb, data):
        r_norm = self.norm_vector(data[:, :, 1])
        h_emb = self._transfer(h_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)
        return - torch.norm(h_emb + r_emb - t_emb, p=self.scoring_fct_norm, dim=-1)

    def _ConvKB(self, h_emb, r_emb, t_emb):
        # print(h_emb.shape)
        bs = h_emb.shape[0]
        # h_emb = h_emb.unsqueeze(1)  # bs x 1 x dim
        h_emb = h_emb.view(-1, h_emb.shape[2]).unsqueeze(1)
        r_emb = r_emb.view(-1, r_emb.shape[2]).unsqueeze(1)
        t_emb = t_emb.view(-1, t_emb.shape[2]).unsqueeze(1)

        # print(h_emb.shape)
        conv_input = torch.cat([h_emb, r_emb, t_emb], 1)  # bs x 3 x dim
        # print(conv_input.shape)
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_size - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)
        score = score.view(bs, -1)
        return score

    def _RotatE(self, h_emb, r_emb, t_emb):
        pi = self.pi_const
        re_head, im_head = torch.chunk(h_emb, 2, dim=2)
        re_tail, im_tail = torch.chunk(t_emb, 2, dim=2)

        phase_relation = r_emb / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.margin.item() - score.sum(dim=2)
        return score

    def _transfer(self, e, norm):
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)
        if e.shape[0] != norm.shape[0]:
            e = e.view(-1, norm.shape[0], e.shape[-1])
            norm = norm.view(-1, norm.shape[0], norm.shape[-1])
            e = e - torch.sum(e * norm, -1, True) * norm
            return e.view(-1, e.shape[-1])
        else:
            return e - torch.sum(e * norm, -1, True) * norm

    def _forward_kg(self, data):
        h_emb = self.e_embedding(data[:, :, 0])
        r_emb = self.r_embedding(data[:, :, 1])
        t_emb = self.e_embedding(data[:, :, 2])
        if self.base_model == 'DistMult':
            return self._DistMult(h_emb, r_emb, t_emb)
        elif self.base_model == 'TransE':
            return self._TransE(h_emb, r_emb, t_emb)
        elif self.base_model == 'TransH':
            return self._TransH(h_emb, r_emb, t_emb, data)
        elif self.base_model == 'ConvKB':
            return self._ConvKB(h_emb, r_emb, t_emb)
        elif self.base_model == 'RotatE':
            return self._RotatE(h_emb, r_emb, t_emb)
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


class Flatten(torch.nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        x = x.view(n, -1)
        return x


def read_data(cfg, fold):
    if cfg.graph == 2:
        train_data_kg = pd.read_csv(cfg.root + '/train_data_kg_G2.csv', index_col=0)
    elif cfg.graph == 1:
        train_data_kg = pd.read_csv(cfg.root + '/train_data_kg_G1.csv', index_col=0)
    else:
        raise ValueError
        
    train_data_tr = pd.read_csv(cfg.root + '/train_data_tr_' + str(fold) + '.csv', index_col=0)
    test_data = pd.read_csv(cfg.root + '/test_data_' + str(fold) + '.csv', index_col=0)

    all_e = set(train_data_kg['h'].unique()) | set(train_data_kg['t'].unique()) | set(
        train_data_tr['tr'].unique()) | set(test_data['tr'].unique())
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
    parser.add_argument('--graph', default=1, type=int)
    # Tunable
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--prior', default=1e-2, type=float)
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--lmbda', default=4, type=float)
    # DistMult, TransE, TransH, ConvKB, RotatE
    parser.add_argument('--base_model', default='DistMult', type=str)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--loss_type', default='pn', type=str)
    parser.add_argument('--scoring_fct_norm', default=1, type=float)   
    parser.add_argument('--kernel_size', default=3, type=int)          
    parser.add_argument('--convkb_drop_prob', default=0.5, type=float)  
    parser.add_argument('--out_channels', default=32, type=int)        
    parser.add_argument('--margin', default=6.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)

    # Untunable
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--valid_interval', default=5, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--tolerance', default=20, type=int)
    return parser.parse_args(args)


def save_model_log(graph_ttl, name):
    with open(name, 'a') as file:
        file.write(graph_ttl)


if __name__ == '__main__':
    cfg = parse_args()
    model_log = 'Configurations:\n'
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
        model_log = model_log + f'\t{arg}: {getattr(cfg, arg)}' + '\n'
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    aucss = []
    auprss = []
    fmaxss = []
    for fold in range(5):
        e_dict, r_dict, train_data_kg, train_data_tr, test_data, already_ts_dict, already_hs_dict = read_data(cfg,
                                                                                                            fold=fold)
        if fold == 0:
            print(f'N Entities:{len(e_dict)}\nN Relations:{len(r_dict)}')
            model_log = model_log + f'N Entities:{len(e_dict)}\nN Relations:{len(r_dict)}' + '\n'
        train_dataset_kg = KnowledgeGraphDataset(e_dict, r_dict, train_data_kg, already_ts_dict, already_hs_dict,
                                                cfg.num_ng)
        train_dataset_tr = TreatmentDataset(train_data_tr)
        test_dataset = TreatmentDataset(test_data)

        train_dataloader_kg = torch.utils.data.DataLoader(dataset=train_dataset_kg,
                                                        batch_size=cfg.bs,
                                                        num_workers=cfg.num_workers,
                                                        shuffle=True,
                                                        drop_last=True)
        train_dataloader_tr = torch.utils.data.DataLoader(dataset=train_dataset_tr,
                                                        batch_size=cfg.bs // 4,
                                                        num_workers=cfg.num_workers,
                                                        shuffle=True,
                                                        drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=cfg.bs // 4,
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
        aucs = []
        auprs = []
        fmaxs = []
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
            avg_loss = round(sum(avg_loss) / len(avg_loss), 6)
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
                f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                fmax = round(np.max(f1_scores), 4)
                th = thresholds[np.argmax(f1_scores)]
                # print(f'AUC:{auc}\tAUPR:{aupr}\tFmax:{fmax}\tTh:{th}')
                aucs.append(auc)
                auprs.append(aupr)
                fmaxs.append(fmax)
                if auc > max_value:
                    max_value = auc
                    tolerance = cfg.tolerance
                else:
                    tolerance -= 1
                if tolerance == 0:
                    break
        max_index = aucs.index(max(aucs))
        aucss.append(aucs[max_index])
        auprss.append(auprs[max_index])
        fmaxss.append(fmaxs[max_index])
        print(
            f'AUC:{aucs[max_index]}\tAUPR:{auprs[max_index]}\tFmax:{fmaxs[max_index]}\tEpoch:{max_index * cfg.valid_interval}')
        model_log = model_log + f'AUC:{aucs[max_index]}\tAUPR:{auprs[max_index]}\tFmax:{fmaxs[max_index]}\tEpoch:{max_index * cfg.valid_interval}' + '\n'
    print(f'AVG# AUC:{round(sum(aucss) / len(aucss), 4)}\tAUPR:{round(sum(auprss) / len(auprss), 4)}\tFmax:{round(sum(fmaxss) / len(fmaxss), 4)}')
    model_log = model_log + f'AVG# AUC:{round(sum(aucss) / len(aucss), 4)}\tAUPR:{round(sum(auprss) / len(auprss), 4)}\tFmax:{round(sum(fmaxss) / len(fmaxss), 4)}' + '\n'
    # save_model_log(model_log, cfg.base_model+'_'+cfg.loss_type+'.log')