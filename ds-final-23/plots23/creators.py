import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from FinalProjDS.plots23.spliters import apply_split_functions, \
    group_by_sections_mean_std

BP_RANGES = ((0, 49), (50, 59), (60, 64), (65, 69), (70, 74), (75, 79),
             (80, 89), (90, 200))
X_BIN_SIZE = 1
Y_BIN_SIZE = 0.05
Y_START = {1: 0.5, 2: 0.85, 3: 0.9, 4: 0.92, 6: 0.95, 8: 0.96, 16: 0.985}


def get_peak_scatter(map_, rate, max_x, length, colorbar_y, colorbar_x=1.1,
                     transpose=False):
    hist, x_edges, y_edges = np.histogram2d(
        map_, rate,
        bins=[
            np.arange(min(map_), max_x + X_BIN_SIZE, X_BIN_SIZE),
            np.arange(min(rate), max(rate) + Y_BIN_SIZE, Y_BIN_SIZE)
        ])
    # Find and plot max map bin for each nor rate
    max_x_bins = np.argmax(hist, axis=0)
    nor_rate_list = []
    map_list = []
    color_list = []
    sum_by_rate = np.sum(hist, axis=0)
    max_sum = 0
    # find the maximum x bin for each y bin
    for i, max_x_bin in enumerate(max_x_bins):
        if sum_by_rate[i] == 0:
            continue
        nor_rate = np.round(y_edges[i], 2)
        map_val = np.round(x_edges[max_x_bin], 2)
        color = hist[max_x_bin][i] * sum_by_rate[i]
        max_sum += color
        nor_rate_list.append(nor_rate)
        map_list.append(map_val)
        color_list.append(color)

    color_list = np.round((color_list / max_sum) * 100, 2)
    hover_text = [f'X: {x_val}<br>Y: {y_val}<br>z: {color_val:.2f}%' for
                  x_val, y_val, color_val in
                  zip(map_list, nor_rate_list, color_list)]
    if transpose:
        map_list, nor_rate_list = nor_rate_list, map_list
    scatter = go.Scatter(x=map_list,
                         y=nor_rate_list,
                         mode='markers',
                         marker=dict(color=color_list,
                                     colorscale='YlGnBu',
                                     colorbar=dict(title='Percentage',
                                                   # thickness=15,
                                                   # len=length,
                                                   # x=colorbar_x,
                                                   # y=colorbar_y
                                                   )),
                         hovertext=hover_text,
                         hoverinfo='text')

    return hist, scatter


# helper function to create plot of nor vs map as heatmap and as scatter plot
# for peaks
def plot_nor_vs_map(df, x_bin_size, y_bin_size, j, rows, max_x):
    map_ = df['cur_bp']
    rate = df['drugrate']
    length = 0.7 / rows
    colorbar_y = Y_START[rows] - j / (rows - 0.5)
    # Compute the 2D histogram using numpy.histogram2d
    hist, peak_scatter = get_peak_scatter(map_, rate, max_x, length,
                                          colorbar_y)

    # Create the go.Histogram2d object
    histogram = go.Histogram2d(
        x=map_,
        y=rate,
        xbins=dict(start=min(map_), end=max(map_) + x_bin_size,
                   size=x_bin_size),
        ybins=dict(start=min(rate), end=max(rate) + y_bin_size,
                   size=y_bin_size),
        z=hist,  # Assign the histogram counts to the 'z' property
        colorscale='Viridis',
        colorbar=dict(
            # thickness=15,
            x=-0.1, )
        # y=colorbar_y, len=length)
    )

    return histogram, peak_scatter


# create plot of nor vs map as heatmap and as scatter plot for peaks for each
# subgroup of the data
def heatmap_and_peak_scatter(df, file_name, split_funcs, max_x,
                             title='', x_bin_size=X_BIN_SIZE,
                             y_bin_size=Y_BIN_SIZE):
    dfs, titles = apply_split_functions(df, split_funcs)
    n = len(titles)
    col1_title = 'Heatmap of MAP VS NOR Rate'
    col2_title = 'MAP Peak Per NOR Rate'
    subplot_titles = [col1_title, col2_title]

    for i in range(n):
        fig = make_subplots(1, 2, subplot_titles=subplot_titles)

        heatmap, scatter = plot_nor_vs_map(dfs[i], x_bin_size, y_bin_size, i,
                                           n, max_x)
        fig.add_trace(heatmap, row=1, col=1)
        fig.add_trace(scatter, row=1, col=2)

        fig.update_xaxes(title_text='MAP')
        fig.update_xaxes(tickmode='linear', dtick=1, range=[55, max_x], col=2)
        fig.update_yaxes(title_text='NOR RATE', tickmode='linear',
                         dtick=Y_BIN_SIZE * 5, col=2)

        fig.update_layout(height=400, width=1000,
                          showlegend=False,
                          # title=dict(x=0.5, y=0.95),
                          # title_text=f'Heatmap of MAP VS NOR RATE With Scatter '
                          #            f'of Peaks:<br>{titles[i]}')
                          )

        fig.write_html(
            f'graphs/heatmaps/html/{max_x}/{file_name}_bin_{Y_BIN_SIZE}_{i}.html')
        fig.write_image(
            f'graphs/heatmaps/img/{max_x}/{file_name}_bin_{Y_BIN_SIZE}_{i}.png')


# Create bar plot of binned MAP with box plot of drugrate for each bin
def box_plot(data, file_name, split_funcs, title):
    dfs, titles = apply_split_functions(data, split_funcs)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.02)

    fig.add_trace(go.Bar(x=titles,
                         y=[len(df) for df in dfs],
                         name='Bin Size'), row=1, col=1)

    # Add box plots for drug rate in each group
    for i, df in enumerate(dfs):
        fig.add_trace(go.Box(y=df['drugrate'],
                             name=f'Drug Rate for MAP {titles[i]}'), row=2,
                      col=1)

    fig.update_layout(xaxis=dict(side='top'),
                      title=title)

    fig.write_html(f'graphs/boxPlots/{file_name}.html')
    fig.write_image(f'graphs/boxPlots/{file_name}.png')


def corr_plot(data, file_name, split_funcs, max_x, title):
    # Create correlation heatmap of MAP and Drugrate for each sub df
    dfs, titles = apply_split_functions(data, split_funcs)
    rows = int(np.ceil(len(titles) / 2))
    cols = 2
    if len(titles) == 1:
        cols = 1
    fig = make_subplots(rows, cols,
                        subplot_titles=titles
                        )
    for i, df in enumerate(dfs):
        df = df[['cur_bp', 'drugrate']]

        correlation_matrix = df.corr()
        cols = ['MAP', 'NOR<br>Rate']
        show = False
        if i == 0:
            show = True
        fig.add_trace(go.Heatmap(z=correlation_matrix.values,
                                 x=cols,
                                 y=cols,
                                 colorscale='Viridis',
                                 showscale=show),
                      row=i // 2 + 1, col=i % 2 + 1)



    # fig.update_layout(title=title)
    fig.write_html(f'graphs/correlation-heatmaps/html/{max_x}/{file_name}.html')
    fig.write_image(f'graphs/correlation-heatmaps/img/{max_x}/{file_name}.png')

# Create scatter plot of data with fitted polynomial of selected degrees
def poly_fit_plot(data, file_name, split_funcs, degrees, title, max_x=None,
                  peak=False, weighted=False):
    if peak:
        get_peak_curve(data, file_name, split_funcs, degrees, title, max_x,
                       weighted)
    else:
        split_funcs.append(group_by_sections_mean_std)
        mean_var_curves(data, file_name, split_funcs, degrees, title)


def get_fit_trace(coefficients, degree, X):
    coeff_str = ""
    for k, coeff in enumerate(coefficients):
        coeff_deg = degree - k
        if coeff_deg == 0:
            coeff_str += f"{coeff:.2f}"
        elif coeff_deg == 1:
            coeff_str += f"{coeff:.2f}x"
        else:
            coeff_str += f"{coeff:.2f}x^{coeff_deg}"
        if k < degree:
            coeff_str += " + "

    # Generate fitted data
    fitted_X = np.linspace(min(X), max(X), 100)
    fitted_y = np.polyval(coefficients, fitted_X)

    fit_trace = go.Scatter(x=fitted_X, y=fitted_y, mode='lines',
                           name=f'Polynomial Degree '
                                f'{degree}:<br>{coeff_str}')
    return fit_trace


def get_peak_curve(data, file_name, split_funcs, degrees, title, max_x,
                   weighted):
    if weighted:
        title += ' Weighted'
        file_name = 'weighted_' + file_name
    dfs, titles = apply_split_functions(data, split_funcs)
    rows = len(titles)
    length = 0.7 / rows

    for i, df in enumerate(dfs):
        peak_fig = go.Figure()
        peak_fig.update_layout(height=400, width=1500)
        y = df['cur_bp'].values.astype(np.float64)
        X = df['drugrate'].values.astype(np.float64)
        colorbar_y = Y_START[rows] - i / (rows - 0.5)
        hist, data_trace = get_peak_scatter(y, X, max_x, length,
                                            colorbar_y, 1, transpose=True)
        data_trace.name = titles[i]
        peak_fig.add_trace(data_trace)

        for degree in degrees:
            # Fit the polynomial
            if weighted:
                weights = data_trace.marker.color
                coefficients = np.polyfit(data_trace.x, data_trace.y, degree,
                                          w=weights)

            else:
                coefficients = np.polyfit(data_trace.x, data_trace.y, degree)

            fit_trace = get_fit_trace(coefficients, degree, data_trace.x)
            peak_fig.add_trace(fit_trace)

        peak_fig.update_yaxes(title_text='MAP', tickmode='linear', dtick=5,
                              range=[55, max_x])
        peak_fig.update_xaxes(title_text='NOR RATE', tickmode='linear',
                              dtick=0.5, )
        peak_fig.update_layout(
            # title=title,
            showlegend=True,
            legend=dict(x=1.1),
        )
        peak_fig.write_html(f'graphs/fitted-curves/html/{max_x}/{file_name}_{i}.html')
        peak_fig.write_image(f'graphs/fitted-curves/img/{max_x}/{file_name}_{i}.png')


def mean_var_curves(data, file_name, split_funcs, degrees, title):
    dfs, titles = apply_split_functions(data, split_funcs)
    rows = len(titles)
    mean_fig = make_subplots(rows, 1, subplot_titles=titles)
    var_fig = make_subplots(rows, 1, subplot_titles=titles)
    full_fig = go.Figure()
    mean_count = 0
    var_count = 0
    for df in dfs:
        type_ = df.iloc[0]['type']
        X = df['cur_bp'].values.astype(np.float64)
        y = df['drugrate'].values.astype(np.float64)
        # Create a scatter plot for the original data
        data_trace = go.Scatter(x=X, y=y,
                                mode='markers', name='data')
        if type_ == 'mean':
            data_trace.name = titles[mean_count]
            mean_count += 1
            mean_fig.add_trace(data_trace, row=mean_count, col=1)

        if type_ == 'var':
            data_trace.name = titles[var_count]
            var_count += 1
            var_fig.add_trace(data_trace, row=var_count, col=1)

        full_fig.add_trace(data_trace)

        if df.shape[0] == 0:
            continue
        for degree in degrees:
            # Fit the polynomial
            coefficients = np.polyfit(X, y, degree)
            fit_trace = get_fit_trace(coefficients, degree, X)
            if type_ == 'mean':
                mean_fig.add_trace(fit_trace, row=mean_count, col=1)

            if type_ == 'var':
                var_fig.add_trace(fit_trace, row=var_count, col=1)

            full_fig.add_trace(fit_trace)

    # Update layout
    full_fig.update_yaxes(title_text='Drug Rate')
    mean_fig.update_yaxes(title_text='Drug Rate')
    var_fig.update_yaxes(title_text='Drug Rate')
    full_fig.update_xaxes(title_text='MAP')
    mean_fig.update_xaxes(title_text='MAP')
    var_fig.update_xaxes(title_text='MAP')

    full_fig.update_layout(
        height=300 * rows, width=2000,
        title="Data with Fitted Curve",
        showlegend=True
    )
    mean_fig.update_layout(
        height=300 * rows, width=1500,
        title="Mean of Data with Fitted Curve",
        showlegend=True
    )
    var_fig.update_layout(
        height=300 * rows, width=1500,
        title="Var Of Data with Fitted Curve",
        showlegend=True
    )

    # Show the plot
    full_fig.write_html(f'graphs/fitted-curves/full_{file_name}.html')
    mean_fig.write_html(f'graphs/fitted-curves/mean_{file_name}.html')
    var_fig.write_html(f'graphs/fitted-curves/var_{file_name}.html')


def create_patient_trajectories(bp, trajectories):
    patient_ids = bp["stay_id"].value_counts()[
        bp["stay_id"].value_counts() > 0].index
    i = 0
    for pat in patient_ids:
        data = bp[bp['stay_id'] == pat]
        # Calculate the threshold value for the top 10 percent
        threshold = data['drugrate'].quantile(0.9)
        # Filter the DataFrame by excluding values above the threshold
        data = data[data['drugrate'] <= threshold]
        i = i + 1
        if i > trajectories:
            break
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        # fig.add_trace(go.Scatter(x=data['cur_bp'], y=data['drugrate'],
        #                          mode='markers+lines', name='MAP VS NOR'),
        #               row=1, col=1, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=data['cur_bp_time'], y=data['cur_bp'], mode='lines',
                       name='MAP Over Time'), secondary_y=False)
        fig.add_trace(
            go.Scatter(x=data['cur_bp_time'], y=data['drugrate'], mode='lines',
                       name='NOR Rate Over Time'),
            secondary_y=True)

        fig.update_layout(title=f'MAP Vs NOR Infusion Rate For StayId: {pat}')

        fig.update_xaxes(title_text='Time')

        fig.update_yaxes(title_text='MAP', secondary_y=False)
        fig.update_yaxes(title_text='NOR RATE', secondary_y=True,
                         showgrid=False)

        fig.write_html(f'graphs/MAP-NOR-PAT/MAP-NOR-{pat}.html')
        fig.write_image(f'graphs/MAP-NOR-PAT/MAP-NOR-{pat}.png')


def drugrate_hist(df, max_x, bin_size):
    day = 24 * 60
    day1 = df[df['cur_bp_time'] < day]['drugrate']
    print(day1.max())
    day2 = df[df['cur_bp_time'] > day]['drugrate']
    fig = go.Figure()
    start = min(min(day1), min(day2))
    end = max(max(day1), max(day2))

    fig.add_trace(go.Histogram(x=day1,
                               xbins=dict(start=start,
                                          end=end,
                                          size=bin_size),
                               name='before 24 hours',
                               histnorm='probability'))
    fig.add_trace(go.Histogram(x=day2,
                               xbins=dict(start=start,
                                          end=end,
                                          size=bin_size),
                               name='after 24 hours',
                               histnorm='probability'))

    fig.update_layout(height=400, width=1000,
                      xaxis=dict(tickangle=45),
                      title=f'Hist With Bins of Size {bin_size} '
                            f'drugrate greater then 0.05')
    fig.write_image(f'graphs/hist/drug_rate_hist_{max_x}_bin_{bin_size}.png')
