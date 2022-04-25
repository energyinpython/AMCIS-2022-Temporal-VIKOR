import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from vikor import VIKOR
from weighting_methods import entropy_weighting
from rank_preferences import rank_preferences
from daria import DARIA
from create_dictionary import Create_dictionary
from correlations import spearman, weighted_spearman

# Function for visualization (bar chart of criteria weights for each evaluated year)

def plot_barplot(df_plot, legend_title):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Criteria', fontsize = 16)
    ax.set_ylabel('Weight value', fontsize = 16)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 16)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 16)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results/' + 'bar_chart_' + legend_title + '_weights.pdf')
    plt.show()



def main():

    # load data
    path = 'dataset'
    weights_type = 'entropy'

    str_years = [str(y) for y in range(2012, 2020)]
    list_alt_names = [r'$A_{' + str(i + 1) + '}$' for i in range(0, 27)]
    preferences = pd.DataFrame(index = list_alt_names)
    rankings = pd.DataFrame(index = list_alt_names)
    results_average = pd.DataFrame(index = list_alt_names)
    df_compare = pd.DataFrame(index = list_alt_names)
    df_presentation = pd.DataFrame(index = list_alt_names)

    averages = np.zeros((27, 10))

    # ====================================================================
    # VIKOR classic approach - for each examined year
    for el, year in enumerate(str_years):
        file = 'data_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        df_data = data.iloc[:len(data) - 1, :]
        types = data.iloc[len(data) - 1, :].to_numpy()
        
        matrix = df_data.to_numpy()
        averages += matrix
        weights = entropy_weighting(matrix)

        if el == 0:
            saved_weights = copy.deepcopy(weights)
        else:
            saved_weights = np.vstack((saved_weights, weights))

        vikor = VIKOR()
        pref = vikor(matrix, weights, types)
        rank = rank_preferences(pref, reverse = False)
        preferences[year] = pref
        rankings[year] = rank
        if year == '2019':
            df_compare['2019'] = rank
            df_presentation['2019 Utility'] = pref
            df_presentation['2019 Rank'] = rank

    list_of_cols_latex = [r'$C_{' + str(i) + '}$' for i in range(1, df_data.shape[1] + 1)]
    df_saved_weights = pd.DataFrame(data = saved_weights, columns = list_of_cols_latex)
    df_saved_weights.index = str_years
    df_saved_weights.index.name = 'Years'
    df_saved_weights.to_csv('results/all_weights.csv')
    plot_barplot(df_saved_weights.T, 'Years')

    preferences = preferences.rename_axis('Ai')
    preferences.to_csv('results/preferences.csv')

    rankings = rankings.rename_axis('Ai')
    rankings.to_csv('results/rankings.csv')

    # =================================================================
    # Display chart of VIKOR rankings for each investigated year
    country_names = list(data.index)
    country_names[4] = country_names[4][:7]
    
    # different colors of lines on chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Dark2(i))
    for i in range(10):
        color.append(cm.tab20b(i))

    ticks = np.arange(1, 28, 2)
    x1 = np.arange(0, len(str_years))
    plt.figure(figsize = (7, 5))
    for i in range(rankings.shape[0]):
        c = color[i]
        plt.plot(x1, rankings.iloc[i, :], color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(list_alt_names[i] + ' ' + country_names[i], (x_max - 0.01, rankings.iloc[i, -1]),
                        fontsize = 13, style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 13)
    plt.ylabel("Rank", fontsize = 13)
    plt.xticks(x1, str_years, fontsize = 13)
    plt.yticks(ticks = ticks, fontsize = 13)
    plt.gca().invert_yaxis()

    plt.grid(True, linestyle = '--', linewidth = 1)
    plt.title('VIKOR rankings', fontsize = 14)
    plt.tight_layout()
    plt.savefig('results/rankings_years.pdf')
    plt.show()

    # ======================================================
    # AVERAGES for average data (benchmarking approach)
    matrix_average = averages / len(str_years)
    weights_average = entropy_weighting(matrix_average)
    pref_average = vikor(matrix_average, weights_average, types)
    rank_average = rank_preferences(pref_average, reverse = False)
    results_average['Pref'] = pref_average
    results_average['Rank'] = rank_average
    df_compare['Average'] = rank_average
    df_presentation['Average Utility'] = pref_average
    df_presentation['Average Rank'] = rank_average
    results_average = results_average.rename_axis('Ai')
    results_average.to_csv('results/average_results.csv')


    # ======================================================================
    # DARIA (DAta vaRIAbility) temporal approach
    # preferences includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = list_alt_names)
    df = preferences.T
    matrix = df.to_numpy()

    # VIKOR orders preferences in ascending order
    met = 'vikor'
    type = -1

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values
    var = daria._std(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, type)

    # for next stage of research
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_results = pd.DataFrame()
    df_results['Ai'] = list(df.columns)
    df_results['Variability'] = list(var)
    
    # list of directions
    df_results['dir list'] = dir_list
    
    df_results.to_csv('results/scores.csv')
    df_varia_fin = df_varia_fin.rename_axis('Ai')
    df_varia_fin.to_csv('results/FINAL.csv')

    # final calculation
    # data with alternatives' rankings' variability values calculated with Gini coeff and directions
    G_df = copy.deepcopy(df_varia_fin)

    # data with alternatives' efficiency of performance calculated for the recent period
    S_df = copy.deepcopy(preferences)

    df_final_results = pd.DataFrame(index = list_alt_names)
    
    S = S_df['2019'].to_numpy()
    G = G_df[met.upper()].to_numpy()
    dir = G_df[met.upper() + ' dir'].to_numpy()

    # update efficiencies using DARIA methodology
    # final updated preferences
    final_S = daria._update_efficiency(S, G, dir)

    # VIKOR has ascending ranking from prefs
    rank = rank_preferences(final_S, reverse = False)
    df_compare['Temporal'] = rank
    df_final_results[met.upper() + ' pref'] = final_S
    df_presentation['Temporal Utility'] = final_S
    df_final_results[met.upper() + ' rank'] = rank
    df_presentation['Temporal Rank'] = rank
    df_presentation['Variability'] = list(var)
    df_presentation['Direction'] = dir_list

    df_final_results.to_csv('results/final_results.csv')
    df_presentation.to_csv('results/presentation.csv')

    df_compare.to_csv('results/compare.csv')

    # ====================================================================
    # correlations
    method_types = list(df_compare.columns)

    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_rs = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(weighted_spearman(df_compare[i], df_compare[j]))
            dict_new_heatmap_rs[j].append(spearman(df_compare[i], df_compare[j]))
        
    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    df_new_heatmap_rw.to_csv('results/df_new_heatmap_rw.csv')
    df_new_heatmap_rs.to_csv('results/df_new_heatmap_rs.csv')


if __name__ == '__main__':
    main()