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
    return ret[ret.max(dim=0)[1][0]].numpy()

if __name__ == '__main__':
    transE = get_best_results('../log/TransE_pn_G1.log')
    rotatE = get_best_results('../log/RotatE_pn_G1.log')
    convkb = get_best_results('../log/ConvKB_pn_G1.log')
    print(f'PN TransE G1:{transE}')
    print(f'PN RotatE G1:{rotatE}')
    print(f'PN ConvKB G1:{convkb}')
    
    pn_G1 = get_best_results('../log/DistMult_pn_G1.log')
    pu_G1 = get_best_results('../log/DistMult_pu_G1.log')
    pu_G2 = get_best_results('../log/DistMult_pu_G2.log')
    pu_G2_extra = get_best_results('../log/DistMult_pu_G2_extra.log')
    print(f'PN DistMult G1:{pn_G1}')
    print(f'PU DistMult G1:{pu_G1}')
    print(f'PU DistMult G2:{pu_G2}')
    print(f'PU DistMult G2 Extra:{pu_G2_extra}')
