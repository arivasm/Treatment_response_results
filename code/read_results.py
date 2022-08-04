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
    pn = get_best_results('../log/DistMult_pn_G2.log')
    pu = get_best_results('../log/DistMult_pu_G2.log')
    print(f'PN DistMult G2:{pn}')
    print(f'PU DistMult G2:{pu}')

