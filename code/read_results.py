import pdb
import torch

def get_best_results(path):
    ret = []
    with open(path) as f:
        for line in f:
            if line[:3] == 'AVG':
                ret_line = []
                for part in line.strip('\n').split('\t'):
                    ret_line.append(float(part.split(':')[-1]))
                ret.append(ret_line)
    ret = torch.tensor(ret)
    return ret[ret.max(dim=0)[1][0]]

if __name__ == '__main__':
    pn_best = get_best_results('./pn.log')
    pu_best = get_best_results('./pu.log')
    print(f'PN DistMult:{pn_best}')
    print(f'PU DistMult:{pu_best}')

