import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import pickle
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult
import pandas as pd
from sklearn.manifold import TSNE


def load_embedding(fold):
    with open('embedding_e_pu_'+fold+'.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def load_rdf_graph(path, name):
    g1 = Graph()
    return g1.parse(path + name, format="ttl")


# === Export results from SPARQL query into a DataFrame ===
def sparql_results_to_df(results: SPARQLResult) -> pd.DataFrame:
    return pd.DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results),
        columns=[str(x) for x in results.vars],
    )


query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example/#> 

select distinct ?treatment ?response
where {
    ?treatment ex:belong_to ?response .
    ?drug ex:part_of ?treatment .
    }
    """


def get_treatment_response(g):
    qres = g.query(query)
    return sparql_results_to_df(qres)


def df_embedding_treatment(treatment_response, dict_emb_e, fold):
    treatment_response.treatment = '<' + treatment_response.treatment + '>'
    emb_treatment = dict((k, dict_emb_e[k]) for k in list(treatment_response.treatment))

    df = pd.DataFrame(emb_treatment.values())
    df['treatment'] = list(emb_treatment.keys())
    df = pd.merge(df, treatment_response, how='inner', on=['treatment'])
    df.response = df.response.str.replace('http://example/#', '')

    train, test = load_train_test(fold)
    df1 = pd.concat([train, test])
    df = pd.merge(df, df1, how='inner', on=['treatment'])
    df.treatment = df.treatment.str.replace('<http://example/Treatment/', '')
    df.treatment = df.treatment.str.replace('>', '')
    return df


import numpy as np
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plot_treatment(new_df, n):
    X = new_df.iloc[:, :-3].copy()
    col = [mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['lightcoral']]
    index = ['effective', 'low_effect']
    # index = ['test', 'train']
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.response.map(color_dictionary)

    #####PLOT#####
    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    # pca = PCA(n_components=2).fit(X)
    # dim_reduction = pca.transform(X)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2500, random_state=42)
    dim_reduction = tsne.fit_transform(X)

    # plt.scatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, marker=new_df.label, s=50)  # alpha=0.6,

    scatter = mscatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, s=25, m=new_df.label)

    # create a list of legend elemntes
    ## markers / records
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
    #                           markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]
    legend_elements = []
    new_df = new_df[['response', 'label', 'c']]
    new_df.drop_duplicates(inplace=True)
    # print(new_df)
    for index, row in new_df.iterrows():
        if row.label == 'o':
            if row.response == 'effective':
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='test_'+row.response,
                                              markerfacecolor=row.c, markersize=8))
            else:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='test_' + row.response,
                                              markerfacecolor=row.c, markersize=8))
        else:
            if row.response == 'effective':
                legend_elements.append(Line2D([0], [0], marker='*', color='w', label='train_'+row.response,
                                              markerfacecolor=row.c, markersize=8))
            else:
                legend_elements.append(Line2D([0], [0], marker='*', color='w', label='train_' + row.response,
                                              markerfacecolor=row.c, markersize=8))

    # plot legend
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    # title and labels
    if n == 1:
        plt.title('Treatments in ' + '${\cal{T\_KG}}_{basic}$', loc='left', fontsize=22)
    elif n == 2:
        plt.title('Treatments in ' + '$\cal{T\_KG}$ + PU', loc='left', fontsize=22)
    else:
        plt.title('Treatments in ' + '${\cal{T\_KG}}_{random}$', loc='left', fontsize=22)
    # plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".pdf", format='pdf', bbox_inches='tight')
    plt.show()


def load_train_test(fold):
    train = pd.read_csv('../data/train_data_tr_' + fold + '.csv')
    test = pd.read_csv('../data/test_data_' + fold + '.csv')
    test.label = 'o'
    test.rename(columns={'tr': 'treatment'}, inplace=True)
    test = test[['treatment', 'label']]

    train.label = '*'
    train.rename(columns={'tr': 'treatment'}, inplace=True)
    train = train[['treatment', 'label']]
    return train, test


if __name__ == '__main__':
    fold = '0'
    dict_emb_e = load_embedding(fold)
    # print(dict_emb_e)

    g = load_rdf_graph('data/T_KG/', 'G1.ttl')
    treatment_response = get_treatment_response(g)

    emb_treatment = df_embedding_treatment(treatment_response, dict_emb_e, fold)
    # print(emb_treatment)
    plot_treatment(emb_treatment, 2)

