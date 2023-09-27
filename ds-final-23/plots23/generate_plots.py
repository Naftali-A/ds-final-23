from FinalProjDS.plots23.spliters import *
import pandas as pd
from FinalProjDS.plots23.creators import *
from sklearn.linear_model import LinearRegression

file_names = {
    'poly_fit': 'poly_fit',
    'heatmap': 'heatmap',
    'corr': 'corr',
    'changes': 'changes',
    'unit': 'unit',
    'norm': 'norm',
    'sal': 'sal/'

}
titles = {
    'poly_fit': 'Map Peak For Each NOR Rate with Fitted Curve',
    'unit': 'By Unit',
    'changes': 'For Change In Rate With Direction',
    'corr': 'Correlation Heatmap',
    'filtered': 'For Filtered: Drugrate (1-50) MAP (30-',
    'norm': '(Normalized NOR Rate)'
}


# Function to remove top 10% of measurements for each patient,
def remove_top_10_percent(group):
    threshold = group['drugrate'].quantile(0.9)
    return group[group['drugrate'] <= threshold]


def filter_drugrate(data, min_, max_):
    return data[(data['drugrate'] < max_) & (data['drugrate'] > min_)]


def filter_map(data, min_, max_):
    return data[(data['cur_bp'] < max_) & (data['cur_bp'] > min_)]


def normalize_weight(df):
    patient_weights = df[['stay_id', 'admissionweight']]
    clean = patient_weights.dropna()
    cleaner = clean.drop_duplicates('stay_id')
    df = df[df['stay_id'].isin(cleaner['stay_id'])]
    df.drop('admissionweight', axis=1, inplace=True)
    df = df.merge(cleaner, on='stay_id', how='left')
    df['drugrate'] /= df['admissionweight']
    np.round(df['drugrate'], 2)
    return df


def generate_sal_plots(df, max_xs, min_y, file_name):
    for max_x in max_xs:
        filtered_drugrate = filter_drugrate(df, min_y, 50)
        filtered_map = filter_map(filtered_drugrate, 30, max_x)
        heatmap_and_peak_scatter(filtered_map,
                                 f"{file_name}_"
                                 f"{file_names['heatmap']}_"
                                 f"{file_names['unit']}_"
                                 f"{file_names['changes']}_"
                                 f"{str(max_x)}",
                                 [
                                     get_bp_by_nor_change_with_direction,
                                     # get_bp_by_unit
                                 ],
                                 max_x,
                                 )

        corr_plot(filtered_map,
                  f"{file_name}_"
                  f"{file_names['corr']}_"
                  f"{file_names['unit']}_"
                  f"{file_names['changes']}_"
                  f"{str(max_x)}",
                  [
                      get_bp_by_nor_change_with_direction,
                      # get_bp_by_unit
                  ],
                  max_x,
                  f"{titles['corr']} "
                  f"{titles['filtered']}{str(max_x)} "
                  f"{titles['unit']} "
                  f"{titles['changes']}")

        corr_plot(filtered_map,
                  f"{file_name}_"
                  f"{file_names['corr']}_"
                  f"{str(max_x)}",
                  [default_spliter],
                  max_x,
                  f"{titles['corr']} "
                  f"{titles['filtered']}{str(max_x)}")


def generate_plots(df, max_xs, normalize, sal):
    min_y = 1
    file_name = ''
    if normalize:
        min_y = 0.05
        df = normalize_weight(df)
        file_name = file_names['norm']
    if sal:
        file_name = f"{file_names['sal']}" + file_name
        generate_sal_plots(df, max_xs, min_y, file_name)
    else:
        for max_x in max_xs:
            filtered_drugrate = filter_drugrate(df, min_y, 50)
            filtered_map = filter_map(filtered_drugrate, 30, max_x)
            # drugrate_hist(filtered_map, max_x, 0.1)
            # drugrate_hist(filtered_map, max_x, 0.05)

            poly_fit_plot(filtered_map,
                          f"{file_name}_"
                          f"{file_names['poly_fit']}_"
                          f"{file_names['unit']}_"
                          f"{file_names['changes']}_"
                          f"{str(max_x)}",
                          [get_bp_by_nor_change_with_direction,
                           get_bp_by_unit],
                          [1, 2, 3],
                          titles['poly_fit'],
                          max_x,
                          peak=True,
                          weighted=False,
                          )
            # poly_fit_plot(filtered_map,
            #               file_names['poly_fit'],
            #               [get_bp_by_nor_change_with_direction,
            #                get_bp_by_unit],
            #               [1],
            #               titles['poly_fit'],
            #               max_x,
            #               peak=True,
            #               weighted=True,
            #               )

            # heatmap_and_peak_scatter(filtered_map,
            #                          f"{file_name}_"
            #                          f"{file_names['heatmap']}_"
            #                          f"{file_names['unit']}_"
            #                          f"{file_names['changes']}_"
            #                          f"{str(max_x)}",
            #                          [
            #                              get_bp_by_nor_change_with_direction,
            #                              get_bp_by_unit
            #                          ],
            #                          max_x,
            #                          )

            # corr_plot(filtered_map,
            #           f"{file_name}_"
            #           f"{file_names['corr']}_"
            #           f"{file_names['unit']}_"
            #           f"{file_names['changes']}_"
            #           f"{str(max_x)}",
            #           [
            #               get_bp_by_nor_change_with_direction,
            #            get_bp_by_unit
            #           ],
            #           max_x,
            #           f"{titles['corr']} "
            #           f"{titles['filtered']}{str(max_x)} "
            #           f"{titles['unit']} "
            #           f"{titles['changes']}")
            #
            # corr_plot(filtered_map,
            #           f"{file_name}_"
            #           f"{file_names['corr']}_"
            #           f"{file_names['unit']}_"
            #           f"{str(max_x)}",
            #           [get_bp_by_unit],
            #           max_x,
            #           f"{titles['corr']} "
            #           f"{titles['filtered']}{str(max_x)} "
            #           f"{titles['unit']}")
            #
            # corr_plot(filtered_map,
            #           f"{file_name}_"
            #           f"{file_names['corr']}_"
            #           f"{str(max_x)}",
            #           [default_spliter],
            #           max_x,
            #           f"{titles['corr']} "
            #           f"{titles['filtered']}{str(max_x)}")


if __name__ == "__main__":
    bp = pd.read_csv('../preprocess/smooth_bp_eicu2.csv')
    # bp['cur_bp'] = bp['otj_filter']
    # print(bp['cur_bp'].isna().sum(), bp.shape)
    bp['drugrate'] = bp['drugrate'].fillna(0)  # fill na with 0
    # generate_plots(bp, [70, 80], normalize=True, sal=False)
    #
    # sal_bp = pd.read_csv('../preprocess/smooth_bp_salz_small.csv')
    # sal_bp['cur_bp'] = sal_bp['otj_filter']
    # sal_bp['drugrate'] = sal_bp['drugrate'].fillna(0)  # fill na with 0
    # generate_plots(sal_bp, [70, 80], normalize=True, sal=True)
    create_patient_trajectories(bp, 10)
