from sklearn.model_selection import KFold
import pandas as pd

def generate_folds(root, graph):
    kg_data = []
    tr_data = []
    with open(root + '/G' + str(graph) + '.ttl') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[1] == 'ex:belong_to':
                if line[2] == 'ex:effective':
                    tr_data.append([line[0], 1])
                elif line[2] == 'ex:low_effect':
                    tr_data.append([line[0], 0])
                else:
                    kg_data.append(line)
            elif line[1] == 'ex:no_belong_to':
                if line[2] == 'ex:low_effect':
                    tr_data.append([line[0], 1])
                elif line[2] == 'ex:effective':
                    tr_data.append([line[0], 0])
                else:
                    kg_data.append(line)
            else:
                kg_data.append(line)

    kg_data = pd.DataFrame(kg_data, columns=['h', 'r', 't'])
    tr_data = pd.DataFrame(tr_data, columns=['tr', 'label']).drop_duplicates()
    kf = KFold(n_splits=5, shuffle=True)
    for i in range(5):
        result = next(kf.split(tr_data), None)
        train_data_tr, test_data = tr_data.iloc[result[0]], tr_data.iloc[result[1]]
        train_data_tr.to_csv(root + '/train_data_tr_' + str(i) + '.csv')
        test_data.to_csv(root + '/test_data_' + str(i) + '.csv')
    kg_data.to_csv(root + '/train_data_kg_G' + str(graph) + '.csv')
    return kg_data

if __name__ == '__main__':
    root = '../data/'
    for i in [1, 2]:
        generate_folds(root, i)
    