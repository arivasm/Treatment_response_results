import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE


def plot_treatment(path, new_df, n, batch):
    X = new_df.iloc[:, :-2].copy()
    col = [mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['lightcoral']]
    index = ['pos', 'neg']
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df['class'].map(color_dictionary)

    # === PLOT ===
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2500, random_state=42)
    dim_reduction = tsne.fit_transform(X)

    plt.scatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, marker='o', s=50)  # alpha=0.6,

    # vocab = list(new_df.triple.values)
    # for i, word in enumerate(vocab):
    #     plt.annotate(word, xy=(dim_reduction[i, 0], dim_reduction[i, 1]))

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]

    # plot legend
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    # title and labels
    if n == 'pn':
        plt.title('Triples in ' + '${\cal{T\_KG}}_{basic}$', loc='left', fontsize=22)
    else:
        plt.title('Triples in ' + '$\cal{T\_KG}$ + PU', loc='left', fontsize=22)

    plt.savefig(fname=path + '2D_' + n + '_' + batch + ".pdf", format='pdf', bbox_inches='tight')
    plt.close()
    # plt.show()


if __name__ == '__main__':
    path = '../data_plot/'
    loss_type = 'pu'
    batch = '0'
    f_name = path + 'embedding_triple_' + loss_type + '_' + batch + '.csv'

    emb_triple = pd.read_csv(f_name)
    print(emb_triple)
    plot_treatment(path, emb_triple, loss_type, batch)
