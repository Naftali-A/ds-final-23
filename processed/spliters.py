import pandas as pd
import numpy as np

BP_RANGES = ((0, 49), (50, 59), (60, 64), (65, 69), (70, 74), (75, 79),
             (80, 89), (90, 200))


# Returns df in array
def default_spliter(df, titles=None):
    return [df], ['All Data']


# Split data into dfs by unit
def get_bp_by_unit(df, titles=None):
    # units = df['unittype_x'].unique()
    units = ['Med-Surg ICU', 'MICU']
    by_unit = [df[df['unittype_x'] == unit] for unit in units]
    if titles:
        units = [unit + ' &<br>' + titles for unit in units]

    return by_unit, units


# Split by bp ranges
def get_bp_by_sections(df, titles=None):
    bins = [f'MAP {m}' for m in BP_RANGES]
    if titles:
        bins = [bin_ + ' &<br>' + titles for bin_ in bins]
    bp_sections = []
    for i, bp_range in enumerate(BP_RANGES):
        bp_low, bp_high = bp_range
        bp_section = df[(df["cur_bp"] >= bp_low) &
                        (df["cur_bp"] <= bp_high)]
        bp_sections.append(bp_section)
    return bp_sections, bins


# Filter rows that had a change in drugrate
def get_bp_by_nor_change(df, titles=None):
    filtered = []
    selected_indices = set()

    # Find the indices where there was a change in the numeric column
    positive_change_indices = df[
        (df['drugrate'] != df['drugrate'].shift())].index
    selected_indices.update(positive_change_indices)

    # Find the indices of the first row for each patient
    first_row_indices = df.groupby('stay_id').head(1).index
    selected_indices.update(first_row_indices)

    filtered.append(df.loc[selected_indices])
    changes = ['Changes in Drugrate']
    if titles:
        changes = [change + '<br>&<br>' + titles for change in changes]

    return filtered, changes


# Filter rows that had a change in drugrate with direction of change
def get_bp_by_nor_change_with_direction(df, titles=None):
    split_df = []

    # Create an empty set to keep track of selected indices
    pos_indices = set()
    neg_indices = set()

    # Find the indices of the first row for each patient
    first_row_indices = df.groupby('stay_id').head(1).index
    pos_indices.update(first_row_indices)

    # Find the indices where there was a positive change in the numeric column
    positive_change_indices = df[
        (df['drugrate'] > df['drugrate'].shift())].index
    pos_indices.update(positive_change_indices)

    # Find the indices where there was a negative change in the numeric column
    negative_change_indices = df[
        (df['drugrate'] < df['drugrate'].shift()) & (
            ~df.index.isin(first_row_indices))].index
    neg_indices.update(negative_change_indices)

    # Create the final DataFrame using the selected indices
    split_df.append(df.loc[pos_indices])
    split_df.append(df.loc[neg_indices])

    changes = ['Positive Changes in Drugrate', 'Negative Changes in Drugrate']
    if titles:
        changes = [change + '<br>&<br>' + titles for change in changes]

    return split_df, changes


def apply_split_functions(data, split_functions):
    dfs = [data]  # Start with the original data
    titles = []
    for split_function in split_functions:
        new_split_dfs = []
        new_split_titles = []
        for i in range(len(dfs)):
            df = dfs[i]
            cur_titles = titles[i] if i < len(titles) else None
            split_df, split_title = split_function(df, cur_titles)
            new_split_dfs.extend(split_df)
            new_split_titles.extend(split_title)

        dfs = new_split_dfs
        titles = new_split_titles

    return dfs, titles


# group by bp ranges and calculate mean and std
def group_by_sections_mean_std(df, titles=None):
    if not titles:
        titles = ['mean', 'var']
    else:
        titles = [titles]
    mean_df = pd.DataFrame(columns=['cur_bp', 'drugrate', 'type'])
    var_df = pd.DataFrame(columns=['cur_bp', 'drugrate', 'type'])
    for bp_range in np.arange(40, 100):
        bp_section = df[df["cur_bp"] == bp_range]
        mean_drugrate = bp_section['drugrate'].mean()
        var_drugrate = bp_section['drugrate'].var()
        if not pd.isna(mean_drugrate):
            mean_df = mean_df.append({
                'cur_bp': bp_range,
                'drugrate': mean_drugrate,
                'type': 'mean',
            }, ignore_index=True)
        if not pd.isna(var_drugrate):
            var_df = var_df.append({
                'cur_bp': bp_range,
                'drugrate': var_drugrate,
                'type': 'var',
            }, ignore_index=True)
    return [mean_df, var_df], titles
