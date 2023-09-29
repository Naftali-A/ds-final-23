import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, t, bartlett, levene, ks_2samp, probplot, ttest_1samp, stats
from itertools import combinations, permutations, product
from sklearn.feature_selection import mutual_info_regression

from sklearn.feature_selection import mutual_info_regression

from plots23.spliters import get_bp_by_nor_change_with_direction


def make_hist(corr, title, cols, dir_name):
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Histogram(x=corr[col], name=col, nbinsx=20))
    # make only 'cur_bp' to be visible
    for i in range(len(fig.data)):
        if fig.data[i].name != 'cur_bp':
            fig.data[i].visible = 'legendonly'
    fig.update_layout(title=title)
    fig.update_xaxes(title='correlation with drugrate')
    try:
        fig.write_html(f'plots/{dir_name}/hist_{title}.html')
        fig.write_image(f'plots/{dir_name}/hist_{title}.png')
    except ValueError:
        pass

    # make now probability histogram of cur_bp
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=corr['cur_bp'], name='cur_bp', nbinsx=20, histnorm='probability'))
    fig.update_layout(title=title)
    # make title bigger
    fig.update_layout(title_font_size=30)
    # make ticks bigger
    fig.update_xaxes(tickfont_size=20)
    fig.update_yaxes(tickfont_size=20)

    try:
        fig.write_html(f'plots/{dir_name}/hist_cur_bp_{title}.html')
        fig.write_image(f'plots/{dir_name}/hist_cur_bp_{title}.png')
    except ValueError:
        pass
    return fig


def make_qq_plots(corr, title, cols, dir_name):
    fig = go.Figure()
    for col in ['cur_bp'] + cols:
        # normalize data
        corr[col] = (corr[col] - corr[col].mean()) / corr[col].std()
        fig.add_trace(go.Scatter(x=probplot(corr[col], dist="norm", plot=None)[0][0],
                                 y=probplot(corr[col], dist="norm", plot=None)[0][1],
                                 mode='markers',
                                 name=col))
    # make only 'cur_bp' to be visible
    # for i in range(len(fig.data)):
    #     if fig.data[i].name != 'cur_bp':
    #         fig.data[i].visible = 'legendonly'
    #         # add x=y line
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='x=y'))
    fig.update_layout(title=title)
    fig.update_xaxes(title='theoretical quantiles')
    fig.update_yaxes(title='sample quantiles')
    fig.write_html(f'plots/{dir_name}/qq_{title}.html')
    fig.write_image(f'plots/{dir_name}/qq_{title}.png')

    # make a new figure with only cur_bp, rolling_2_mean, rolling_3_mean, rolling_4_mean and rolling_2_median, rolling_3_median, rolling_4_median
    fig = go.Figure()
    for col in ['cur_bp', 'rolling_2_mean', 'rolling_3_mean', 'rolling_4_mean', 'rolling_2_median', 'rolling_3_median',
                'rolling_4_median']:
        fig.add_trace(go.Scatter(x=probplot(corr[col], dist="norm", plot=None)[0][0],
                                 y=probplot(corr[col], dist="norm", plot=None)[0][1],
                                 mode='markers',
                                 name=col))
    # add x=y
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='x=y'))
    # make the title bigger
    # fig.update_layout(title=title)
    # fig.update_layout(title_font_size=30)

    fig.update_xaxes(title='theoretical quantiles', tickfont_size=25)
    fig.update_yaxes(title='sample quantiles', tickfont_size=25)
    fig.update_xaxes(tickfont_size=15)
    fig.update_yaxes(tickfont_size=15)
    # make legend font bigger
    fig.update_layout(legend_font_size=15)
    fig.write_html(f'plots/{dir_name}/qq_cur_bp_{title}.html')
    fig.write_image(f'plots/{dir_name}/qq_cur_bp_{title}.png')

    return fig


def apply_tests_over_cols_and_type(corr, types, my_cols, dir_name):
    shapiro_df = pd.DataFrame(columns=types, index=my_cols)
    t_test_df = pd.DataFrame(columns=types, index=my_cols)
    for col, unit_type in product(my_cols, types):
        x = corr.loc[
            (corr.index.get_level_values(0) == unit_type) & (
                    corr.index.get_level_values(2) == 'drugrate'), col].reset_index(level=2, drop=True)
        try:
            shapiro_df.loc[col, unit_type] = shapiro(x.dropna()).pvalue
            t_test_df.loc[col, unit_type] = ttest_1samp(x.dropna(), 0)[1]
        except ValueError:
            pass

    fig = go.Figure()
    # make heatmap for shapiro p-values
    fig.add_trace(go.Heatmap(z=shapiro_df, x=shapiro_df.columns, y=shapiro_df.index, name='shapiro p-values'))
    fig.update_layout(title='shapiro p-values')
    fig.write_html(f'plots/{dir_name}/shapiro_p_values.html')
    fig.write_image(f'plots/{dir_name}/shapiro_p_values.png')

    fig = go.Figure()
    # make heatmap for shapiro results with alpha=0.05
    fig.add_trace(go.Heatmap(z=(shapiro_df < 0.05).astype(int), x=shapiro_df.columns, y=shapiro_df.index,
                             name='shapiro results with alpha=0.05'))
    fig.update_layout(title='shapiro results with alpha=0.05')
    fig.write_html(f'plots/{dir_name}/shapiro_results.html')
    fig.write_image(f'plots/{dir_name}/shapiro_results.png')

    fig = go.Figure()
    # make heatmap for t-test p-values
    fig.add_trace(go.Heatmap(z=t_test_df, x=t_test_df.columns, y=t_test_df.index, name='t-test p-values'))
    fig.update_layout(title='t-test p-values')
    fig.write_html(f'plots/{dir_name}/t_test_p_values.html')
    fig.write_image(f'plots/{dir_name}/t_test_p_values.png')

    fig = go.Figure()
    # make heatmap for t-test results with alpha=0.05
    fig.add_trace(
        go.Heatmap(z=(t_test_df < 0.05).astype(int), x=t_test_df.columns, y=t_test_df.index,
                   name='t-test results with alpha=0.05'))
    fig.update_layout(title='t-test results with alpha=0.05')
    fig.write_html(f'plots/{dir_name}/t_test_results.html')
    fig.write_image(f'plots/{dir_name}/t_test_results.png')


def make_hists_and_qq_plots(bp_corr, unit_types, unit_names, cols, dir_name):
    for unit_type, unit_name in zip(unit_types, unit_names):
        bp_corr_by_type = bp_corr.loc[
            (bp_corr.index.get_level_values(0).isin(unit_type)) & (
                    bp_corr.index.get_level_values(2) == 'drugrate'), cols + [
                'cur_bp']].reset_index(level=2, drop=True)

        make_hist(bp_corr_by_type, unit_name, ['cur_bp'] + cols, dir_name)
        make_qq_plots(bp_corr_by_type, unit_name, cols, dir_name)

    fig = go.Figure()
    salz_cur_bp_mean = bp_corr.loc[(bp_corr.index.get_level_values(0) == 'Salzburg') & (
            bp_corr.index.get_level_values(2) == 'drugrate'), 'cur_bp'].mean()
    eICU_cur_bp_mean = bp_corr.loc[(bp_corr.index.get_level_values(0) == 'eICU') & (
            bp_corr.index.get_level_values(2) == 'drugrate'), 'cur_bp'].mean()
    for statistic_name in ['mean', 'median', 'min', 'max']:
        # choose a specific color for each statistic
        color = 'red' if statistic_name == 'mean' else 'blue' if statistic_name == 'median' else 'green' if statistic_name == 'min' else 'orange'
        # find all cols that contain the statistic name
        cols_with_statistic = [col for col in cols if statistic_name in col]
        # find mean of all column
        salz_means = bp_corr.loc[(bp_corr.index.get_level_values(0) == 'Salzburg') & (
                bp_corr.index.get_level_values(2) == 'drugrate'), ['cur_bp'] + cols_with_statistic].mean()
        eICU_means = bp_corr.loc[(bp_corr.index.get_level_values(0) == 'eICU') & (
                bp_corr.index.get_level_values(2) == 'drugrate'), ['cur_bp'] + cols_with_statistic].mean()
        x = [i for i in range(1, 11)]

        fig.add_trace(go.Scatter(x=x, y=salz_means, name=f'Salzburg {statistic_name}', mode='markers', marker=dict(color=color, symbol='square')))
        fig.add_trace(go.Scatter(x=x, y=eICU_means, name=f'eICU {statistic_name}', mode='markers', marker=dict(color=color)))

        # add ls line
        salz_slope, salz_intercept, salz_r_value, salz_p_value, salz_std_err = stats.linregress(x, salz_means)
        eICU_slope, eICU_intercept, eICU_r_value, eICU_p_value, eICU_std_err = stats.linregress(x, eICU_means)
        fig.add_trace(go.Scatter(x=x, y=[salz_slope * i + salz_intercept for i in x], name=f'Salzburg {statistic_name} ls',
                                    mode='lines', line=dict(color=color), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[eICU_slope * i + eICU_intercept for i in x], name=f'eICU {statistic_name} ls',
                                    mode='lines', line=dict(color=color), showlegend=False))



    fig.add_trace(go.Scatter(x=x, y=[salz_cur_bp_mean] * 10, name='Salzburg MAP', mode='markers'))
    fig.add_trace(go.Scatter(x=x, y=[eICU_cur_bp_mean] * 10, name='eICU MAP', mode='markers'))
    fig.update_layout(xaxis_title='rolling parameter', yaxis_title='mean of correlation statistic')
    fig.write_html(f'plots/{dir_name}/means_per_unit.html')
    fig.write_image(f'plots/{dir_name}/means_per_unit.png')
    fig.show()



def apply_3_way_tests_over_cols(corr, param, my_cols, dir_name):
    p_value_3_way_tests = pd.DataFrame(index=my_cols, columns=['bartlett_p_value', 'bartlett_result', 'levene_p_value',
                                                               'levene_result'])

    for col in ['cur_bp'] + cols:
        micu_corr = corr.loc[
            (corr.index.get_level_values(0) == 'MICU') & (corr.index.get_level_values(2) == 'drugrate'), col].dropna()
        sicu_corr = corr.loc[
            (corr.index.get_level_values(0) == 'SICU') & (corr.index.get_level_values(2) == 'drugrate'), col].dropna()
        medsurg_corr = corr.loc[
            (corr.index.get_level_values(0) == 'Med-Surg ICU') & (
                    corr.index.get_level_values(2) == 'drugrate'), col].dropna()
        p_value_3_way_tests.loc[col, 'bartlett_p_value'] = bartlett(micu_corr, sicu_corr, medsurg_corr)[1]
        p_value_3_way_tests.loc[col, 'bartlett_result'] = p_value_3_way_tests.loc[col, 'bartlett_p_value'] < 0.05
        p_value_3_way_tests.loc[col, 'levene_p_value'] = levene(micu_corr, sicu_corr, medsurg_corr)[1]
        p_value_3_way_tests.loc[col, 'levene_result'] = p_value_3_way_tests.loc[col, 'levene_p_value'] < 0.05

    fig = go.Figure()
    # make heatmap for table
    fig.add_trace(
        go.Heatmap(z=p_value_3_way_tests.astype(float), x=p_value_3_way_tests.columns, y=p_value_3_way_tests.index,
                   name='3-way tests p-values'))
    fig.update_layout(title='3-way tests p-values')
    fig.write_html(f'plots/{dir_name}/3_way_tests_p_values.html')
    fig.write_image(f'plots/{dir_name}/3_way_tests_p_values.png')


def var_ratio_perm_test(x, y, perm_num=1000):
    x = x ** 2
    y = y ** 2
    var_ratio = np.mean(x) / np.mean(y)
    if var_ratio < 1:
        var_ratio = 1 / var_ratio
        x, y = y, x
    # concatenate x and y
    concat_data = np.concatenate([x, y])
    permutations = np.array([np.random.permutation(concat_data) for _ in range(perm_num)])
    # calculate var ratio for each permutation
    perm_var_ratios = np.mean(permutations[:, :len(x)], axis=1) / np.mean(permutations[:, len(x):], axis=1)
    # calculate p value
    # print(perm_var_ratios)
    p_value = np.sum(perm_var_ratios >= var_ratio) / perm_num
    return None, p_value


def apply_tests_over_unit_types_and_every_2_cols(corr, u_types, u_names, my_cols, dir_name):
    index = pd.MultiIndex.from_product([u_types, my_cols, ['bartlett', 'levene', 'ks_2samp', 'var_ratio_perm_test']],
                                       names=['unit_type', 'col', 'test'])
    p_values_between_differnt_cols = pd.DataFrame(index=index, columns=my_cols)

    for unit_type, unit_name in zip(u_types, u_names):
        corr2 = corr.loc[
            (corr.index.get_level_values(0) == unit_type) & (
                    corr.index.get_level_values(2) == 'drugrate'), my_cols].reset_index(level=2, drop=True)
        for col1, col2 in combinations(my_cols, 2):
            for test, test_name in [(bartlett, 'bartlett'), (levene, 'levene'), (ks_2samp, 'ks_2samp'),
                                    (var_ratio_perm_test, 'var_ratio_perm_test')]:
                try:
                    p_values_between_differnt_cols.loc[(unit_type, col1, test_name), col2] = \
                        test(corr2[col1].dropna(), corr2[col2].dropna())[1]
                    p_values_between_differnt_cols.loc[(unit_type, col2, test_name), col1] = \
                        p_values_between_differnt_cols.loc[
                            (unit_type, col1, test_name), col2]
                except ValueError:
                    pass
        for test, test_name in [(bartlett, 'bartlett'), (levene, 'levene'), (ks_2samp, 'ks_2samp'),
                                (var_ratio_perm_test, 'var_ratio_perm_test')]:
            fig = go.Figure()
            # make heatmap for table
            fig.add_trace(
                go.Heatmap(z=p_values_between_differnt_cols.loc[(unit_type, slice(None), test_name), :].astype(float),
                           x=p_values_between_differnt_cols.columns,
                           y=p_values_between_differnt_cols.loc[(unit_type, slice(None), test_name),
                             :].index.get_level_values(1),
                           name=f'{test.__name__} p-values'))
            fig.update_layout(title=f'{test.__name__} p-values for {unit_name}')
            fig.write_html(f'plots/{dir_name}/{test.__name__}_p_values_for_{unit_name}.html')
            fig.write_image(f'plots/{dir_name}/{test.__name__}_p_values_for_{unit_name}.png')

            fig = go.Figure()
            # make heatmap for decision with alpha=0.05
            fig.add_trace(go.Heatmap(
                z=(p_values_between_differnt_cols.loc[(unit_type, slice(None), test_name), :] < 0.05).astype(int),
                x=p_values_between_differnt_cols.columns,
                y=p_values_between_differnt_cols.loc[(unit_type, slice(None), test_name), :].index.get_level_values(1),
                name=f'{test.__name__} decision'))

            fig.update_layout(title=f'{test.__name__} decision for {unit_name} with alpha=0.05')
            fig.write_html(f'plots/{dir_name}/{test.__name__}_decision_for_{unit_name}_with_alpha_0.05.html')
            fig.write_image(f'plots/{dir_name}/{test.__name__}_decision_for_{unit_name}_with_alpha_0.05.png')


def do_corr_tests(corr, u_types, u_names, my_cols, dir_name, bp_col_name='cur_bp'):
    make_hists_and_qq_plots(corr, u_types, u_names, my_cols, dir_name)

    apply_tests_over_cols_and_type(corr, u_names, [bp_col_name] + my_cols, dir_name)

    apply_3_way_tests_over_cols(corr, u_names, [bp_col_name] + my_cols, dir_name)

    apply_tests_over_unit_types_and_every_2_cols(corr, u_names, u_names, [bp_col_name] + my_cols, dir_name)


def make_mi_table(bp_by_type, cols):
    my_by_type = bp_by_type.filter(lambda x: x.isna().any(axis=1))
    drop rows with nan
    my_by_type = my_by_type.dropna()
    print(my_by_type)

    def get_mi_wrapper(x):
        try:
            x = x[cols + ['drugrate']].dropna()
            x = mutual_info_regression(x[cols], x['drugrate'], discrete_features=True)
            print(x)
            return x
        except ValueError:
            return [np.nan] * len(cols)

    p = bp_by_type.apply(get_mi_wrapper)
    # every cell in p is a list of len(cols) with the mi values, so we need to convert it to a dataframe
    p = pd.DataFrame(p.values.tolist(), index=pd.MultiIndex.from_tuples([(x, y, 'drugrate') for x, y in p.index]),
                     columns=cols)
    # add another layer to the index with a single group 'drugrate'
    print(p.index)
    return p


def corr_tests_flow(bp, dir_name_prefix="",
                    unit_types=None,
                    unit_names=None):
    if unit_names is None:
        unit_names = ['MICU', 'SICU', 'Med-Surg ICU']
    if unit_types is None:
        unit_types = [['MICU'], ['SICU'], ['Med-Surg ICU'], ['SICU', 'MICU', 'Med-Surg ICU']]
    bp_by_type = bp.loc[bp['unittype_x'].isin(unit_names)].groupby(['unittype_x', 'stay_id'])

    bp_corr = bp_by_type.corr()
    if not os.path.exists(f'plots/{dir_name_prefix}whole_population'):
        os.makedirs(f'plots/{dir_name_prefix}whole_population')
    do_corr_tests(bp_corr, unit_types, unit_names, cols, dir_name_prefix + 'whole_population')

    bp_with_change, _ = get_bp_by_nor_change_with_direction(bp)
    # unite the two dataframes
    bp_with_change = pd.concat(bp_with_change)
    bp_with_change_corr = bp_with_change.groupby(['unittype_x', 'stay_id']).corr()
    if not os.path.exists(f'plots/{dir_name_prefix}whole_population_with_change'):
        os.makedirs(f'plots/{dir_name_prefix}whole_population_with_change')
    do_corr_tests(bp_with_change_corr, unit_types, unit_names, cols, dir_name_prefix + 'whole_population_with_change')

    bin_size = 5
    bins = [[bp, bp + bin_size] for bp in range(40, 120, bin_size)] + \
           [[bp, bp + bin_size * 2] for bp in range(40, 120, bin_size * 2)] + \
           [[bp, bp + bin_size * 4] for bp in range(40, 120, bin_size * 4)]

    for my_bin in bins:
        bins_bp = bp.loc[(bp['cur_bp'] >= my_bin[0]) & (bp['cur_bp'] < my_bin[1])]
        bins_bp_by_type = bins_bp.groupby(['unittype_x', 'stay_id'])
        bins_bp_corr = bins_bp_by_type.corr()
        # if there is no dir for this bin, make it
        dir_name = f'bin_{my_bin[0]}_{my_bin[1]}'
        if not os.path.exists(f'plots/{dir_name}'):
            os.makedirs(f'plots/{dir_name}')
        do_corr_tests(bins_bp_corr, unit_types, unit_names, cols, dir_name)

        bins_bp_with_change, _ = get_bp_by_nor_change_with_direction(bins_bp)
        bins_bp_with_change = pd.concat(bins_bp_with_change)
        bins_bp_with_change_corr = bins_bp_with_change.groupby(['unittype_x', 'stay_id']).corr()
        # if there is no dir for this bin, make it
        dir_name = f'bin_{my_bin[0]}_{my_bin[1]}_with_change'
        if not os.path.exists(f'plots/{dir_name}'):
            os.makedirs(f'plots/{dir_name}')
        do_corr_tests(bins_bp_with_change_corr, unit_types, unit_names, cols, dir_name)



    bp_mi = make_mi_table(bp_by_type, ['cur_bp'] + cols)
    if not os.path.exists(f'plots/whole_population_mi'):
        os.makedirs(f'plots/whole_population_mi')
    do_corr_tests(bp_mi, unit_types, unit_names, cols, 'whole_population_mi')

    for my_bin in bins:
        bins_bp = bp.loc[(bp['cur_bp'] >= my_bin[0]) & (bp['cur_bp'] < my_bin[1])]
        bins_bp_by_type = bins_bp.groupby(['unittype_x', 'stay_id'])
        bins_bp_mi = make_mi_table(bins_bp_by_type, ['cur_bp'] + cols)
        dir_name = f'bin_{my_bin[0]}_{my_bin[1]}_mi'
        if not os.path.exists(f'plots/{dir_name}'):
            os.makedirs(f'plots/{dir_name}')
        do_corr_tests(bins_bp_mi, unit_types, unit_names, cols, dir_name)


if __name__ == '__main__':
    eicu = pd.read_csv("../preprocess/smooth_bp_eicu2.csv")
    eicu = eicu.loc[eicu['unittype_x'].isin(['SICU', 'MICU', 'Med-Surg ICU']), :]
    eicu_corr = eicu.groupby(['unittype_x', 'stay_id']).corr()
    corr_tests_flow(eicu_corr)

    eicu.unittype_x = "eICU"

    salz = pd.read_csv("../preprocess/smooth_bp_salz.csv")
    salz['unittype_x'] = "Salzburg"

    bp = pd.concat([eicu, salz])
    cols = []
    for i in range(2, 11):
        cols.append(f'rolling_{i}_mean')
        # cols.append(f'rolling_{i}_std')
        cols.append(f'rolling_{i}_min')
        cols.append(f'rolling_{i}_max')
        cols.append(f'rolling_{i}_median')

    cols_otj = []
    for i in range(2, 11):
        cols_otj.append(f'rolling_{i}_mean_otj')
        cols_otj.append(f'rolling_{i}_min_otj')
        cols_otj.append(f'rolling_{i}_max_otj')
        cols_otj.append(f'rolling_{i}_median_otj')

    bp = bp[['drugrate', 'cur_bp_time', 'stay_id', 'cur_bp', 'unittype_x', 'otj_filter'] + cols + cols_otj]
    # drop all the columns that are in cols
    bp = bp.drop(columns=cols)
    # if a column ends with _otj, remove the suffix
    bp = bp.rename(columns={x: x[:-4] for x in cols_otj if x.endswith('_otj')})
    # drop cur_bp
    bp = bp.drop(columns=['cur_bp'])
    # change otj_filter to 'cur_bp'
    bp = bp.rename(columns={'otj_filter': 'cur_bp'})
    corr_tests_flow(bp, dir_name_prefix="country_", unit_types=[["eICU"], ["Salzburg"]],
                    unit_names=["eICU", "Salzburg"])
