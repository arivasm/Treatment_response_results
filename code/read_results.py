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
    DistMult_pn = get_best_results('./DistMult_pn.log')
    DistMult_pu = get_best_results('./DistMult_pu.log')
    DistMult_pur = get_best_results('./DistMult_pur.log')
    print(f'PN DistMult:{DistMult_pn}')
    print(f'PU DistMult:{DistMult_pu}')
    print(f'PUR DistMult:{DistMult_pur}')
    TransE_pn = get_best_results('./TransE_pn.log')
    TransE_pu = get_best_results('./TransE_pu.log')
    TransE_pur = get_best_results('./TransE_pur.log')
    print(f'PN TransE:{TransE_pn}')
    print(f'PU TransE:{TransE_pu}')
    print(f'PUR TransE:{TransE_pur}')

