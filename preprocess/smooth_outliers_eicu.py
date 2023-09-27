import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_bp(num_of_pats=300, bp_max=160, bp_min=30):
    drug = pd.read_csv("../data/eICU/infusiondrug.csv", low_memory=False)
    diagnosis = pd.read_csv("../data/eICU/diagnosis.csv")
    patients_weight = pd.read_csv("../data/eICU/patient.csv")[["patientunitstayid", "admissionweight"]]
    patient = pd.read_csv("../data/eICU/patient.csv")
    drug = drug.merge(patients_weight, on="patientunitstayid")

    # add hospitalID and unitType and wardID to the drug table corelated to patientunitstayid
    drug["hospitalid"] = drug["patientunitstayid"].map(patient.set_index("patientunitstayid")["hospitalid"])
    drug["unittype"] = drug["patientunitstayid"].map(patient.set_index("patientunitstayid")["unittype"])
    drug["wardid"] = drug["patientunitstayid"].map(patient.set_index("patientunitstayid")["wardid"])

    # filter drug from NA in drugname, patientunitstayid, hospitalid, unittype, wardid
    drug = drug[~drug['drugname'].isna()]
    drug = drug[~drug['patientunitstayid'].isna()]
    drug = drug[~drug['hospitalid'].isna()]
    drug = drug[~drug['unittype'].isna()]
    drug = drug[~drug['wardid'].isna()]

    # filter drugs with other drug names
    other_drugs = drug[drug['drugname'].str.startswith(("Epinephrine", "Dopamine", "Vesopressin"))]

    # filter drug to include only drugnames that start with "Norepinephrine"
    drug = drug[drug['drugname'].str.startswith("Norepinephrine")]

    pat_at_stage['with NOR'] = drug['patientunitstayid'].nunique()

    # remove patients who recieved other vasopressors
    drug = drug[~drug['patientunitstayid'].isin(other_drugs['patientunitstayid'])]

    # sort drug by patientunitstayid and drugstartoffset
    drug = drug.sort_values(by=["patientunitstayid", "infusionoffset"])

    pat_at_stage['with only NOR'] = drug['patientunitstayid'].nunique()

    bp = pd.read_csv("../preprocess/filtered_bp_eicu.csv")
    bp = bp[bp["cur_bp"] < bp_max]
    bp = bp[bp["cur_bp"] > bp_min]
    bp = bp.sort_values(by=["stay_id", "cur_bp_time"])
    bp = bp[bp["stay_id"].isin(drug["patientunitstayid"])]
    bp["next_bp_time"] = bp.groupby("stay_id")["cur_bp_time"].shift(-1)
    bp["interval"] = bp["next_bp_time"] - bp["cur_bp_time"]
    drug["drugrate"] = pd.to_numeric(drug["drugrate"], errors='coerce')
    bp["age"] = pd.to_numeric(bp["stay_id"].map(patient.set_index("patientunitstayid")["age"]), errors='coerce')
    bp["hospitalid"] = bp["stay_id"].map(patient.set_index("patientunitstayid")["hospitalid"])
    bp["unittype"] = bp["stay_id"].map(patient.set_index("patientunitstayid")["unittype"])
    bp["wardid"] = bp["stay_id"].map(patient.set_index("patientunitstayid")["wardid"])

    bp["hospitalDischargeStatus"] = bp["stay_id"].map(
        patient.set_index("patientunitstayid")["hospitaldischargestatus"]).map({"Alive": 0, "Expired": 1})

    # find the number of patients per hospital
    bp_hosp = bp.groupby(["hospitalid"]).agg({"stay_id": ["nunique"]}).sort_values(by=("stay_id", "nunique"))

    bp['num_of_pats'] = bp['hospitalid'].map(bp_hosp[('stay_id', 'nunique')])

    # bp_big will consist of patients from hospitalid with more then 400 differnt patientsunitstayid
    bp_big = bp[bp["hospitalid"].isin(drug.groupby("hospitalid")["patientunitstayid"].nunique().sort_values()[
                                          drug.groupby("hospitalid")[
                                              "patientunitstayid"].nunique().sort_values() > num_of_pats].index)]
    # filter sru to contain only patients from bp_big
    drug = drug[drug["patientunitstayid"].isin(bp_big["stay_id"])]
    drug.to_csv('filtered_drug_eicu.csv', index=False)
    drug.rename(columns={'infusionoffset': 'cur_bp_time', 'patientunitstayid': 'stay_id'}, inplace=True)
    bp_big = pd.merge(bp_big, drug, on=['stay_id', 'cur_bp_time'], how='outer')
    bp_big = bp_big.sort_values(by=["stay_id", "cur_bp_time"])
    bp_big['drugrate'] = bp_big.groupby('stay_id')['drugrate'].transform(lambda x: x.ffill())
    bp_big['infusionrate'] = bp_big.groupby('stay_id')['infusionrate'].transform(lambda x: x.ffill())
    bp_big['drugamount'] = bp_big.groupby('stay_id')['drugamount'].transform(lambda x: x.ffill())
    bp_big = bp_big[~bp_big["cur_bp"].isna()]

    return bp_big


def smooth_outliers(big_bp: pd.DataFrame, threshold_constant: float = 3):
    for i in [3, 5, 10, 15, 20]:
        big_bp["rolling"] = big_bp.groupby("stay_id")["cur_bp"].rolling(i, center=True).mean().reset_index(0,
                                                                                                           drop=True)
        # replace NaN with the original value
        big_bp["rolling"] = big_bp["rolling"].fillna(big_bp["cur_bp"])
        big_bp["rolling_res"] = big_bp["cur_bp"] - big_bp["rolling"]
        big_bp["rolling_res"] = big_bp["rolling_res"].abs()

        # add column with the median of the residuals
        big_bp["rolling_res_median"] = big_bp.groupby("stay_id")["rolling_res"].transform(
            "median")
        # add column with outliers removed (outliers are defined as residuals that are 10 times the median)
        big_bp["rolling_res_median"] = big_bp["rolling_res_median"] * threshold_constant
        big_bp["smooth_" + str(i)] = big_bp["cur_bp"]
        big_bp.loc[big_bp["rolling_res"] > big_bp["rolling_res_median"], "smooth_" + str(
            i)] = big_bp["rolling"]

        # replace NaN with the original value
        big_bp["rolling"] = big_bp["rolling"].fillna(big_bp["otj_filter"])
        big_bp["rolling_res"] = big_bp["otj_filter"] - big_bp["rolling"]
        big_bp["rolling_res"] = big_bp["rolling_res"].abs()

        # add column with the median of the residuals
        big_bp["rolling_res_median"] = big_bp.groupby("stay_id")[
            "rolling_res"].transform(
            "median")
        # add column with outliers removed (outliers are defined as residuals that are 10 times the median)
        big_bp["rolling_res_median"] = big_bp["rolling_res_median"] * threshold_constant
        big_bp["smooth_otj_" + str(i)] = big_bp["otj_filter"]
        big_bp.loc[
            big_bp["rolling_res"] > big_bp["rolling_res_median"], "smooth_otj_" + str(
                i)] = big_bp["rolling"]
    return big_bp


def remove_one_time_jumps(bp_df, jump_threshold=15, distance_threshold=15):
    # filter pat's bp from values that are have jumps > 15 and are close to the x->-x line
    bp_df.loc[:, 'otj_filter'] = bp_df.loc[:, 'cur_bp']

    bp_df.loc[bp_df['interval'] > 5, 'otj_filter'] = np.nan

    jumps = np.r_[bp_df['otj_filter'][1:].values - bp_df['otj_filter'][:-1].values, [np.nan]]
    bi_jump = np.c_[jumps[:-1], jumps[1:]]
    bi_jump = np.r_[[[np.nan, np.nan]], bi_jump]
    # project bi_jump on x->-x line
    proj_bi_jump = np.outer([np.sqrt(1 / 2), -np.sqrt(1 / 2)], [np.sqrt(1 / 2), -np.sqrt(1 / 2)]) @ bi_jump.T

    distances = np.linalg.norm(bi_jump - proj_bi_jump.T, axis=1)

    bp_df.loc[(np.abs(bi_jump[:, 0]) > jump_threshold) & (distances < distance_threshold), 'otj_filter'] = np.nan

    params = {
        (15, 15),
        (15, 10),
        (10, 15),
        (10, 10),
        (20, 20),
        (20, 15),
        (15, 20),
        (20, 10),
        (10, 20),
    }
    for j_threshold, d_threshold in params:
        bp_df.loc[:, f'otj_filter_{j_threshold}_{d_threshold}'] = bp_df.loc[:, 'cur_bp']
        bp_df.loc[(np.abs(bi_jump[:, 0]) > j_threshold) & (
                distances < d_threshold), f'otj_filter_{j_threshold}_{d_threshold}'] = np.nan
    bp_df['distances'] = distances
    bp_df['jump'] = jumps
    return bp_df


def smooth_with_rolling_gaussian_proccess(bp_df, sigma=2, length_scale=1, alpha=1, window_size=10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, PairwiseKernel, ConstantKernel, WhiteKernel

    s = None

    def rolling_gp(x: pd.DataFrame, kernel, alpha=1):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
        # import polynomial fitting from sklearn
        from sklearn.preprocessing import PolynomialFeatures
        # import linear regression from sklearn
        from sklearn.linear_model import LinearRegression
        # import pipeline from sklearn
        from sklearn.pipeline import Pipeline
        # create a pipeline that first transforms the data with polynomial features
        # and then fits a linear regression model
        gp = Pipeline([('polynomial_features', PolynomialFeatures(degree=2)),
                       ('linear_regression', LinearRegression())])

        def gp_with_dropna(y: pd.DataFrame):
            ynna = y.dropna()
            if len(ynna) == 0:
                return y[s]
            gp.fit(ynna['cur_bp_time'].values.reshape(-1, 1), ynna['otj_filter'])
            y[s] = gp.predict(y['cur_bp_time'].values.reshape(-1, 1))
            return y[s]

        pp = x.groupby(x.index // window_size).apply(gp_with_dropna).reset_index(0, drop=True)
        x[s] = pp
        return x[s]

    sigma, alpha, l = 0.1, 0.1, 0.1
    s = f'gp_sigma-{sigma}_l-{l}_alp-{alpha}'
    bp_df[s] = 0
    k = bp_df[['stay_id', 'cur_bp_time', 'otj_filter', s]].groupby(['stay_id']).apply(
        lambda x: pd.Series(rolling_gp(x, kernel=sigma * RBF(sigma),
                                       alpha=alpha)))
    bp_df['gp'] = k.reset_index(0, drop=True)

    sigma, alpha, l = 0.1, 0.1, 10
    s = f'gp_sigma-{sigma}_l-{l}_alp-{alpha}'
    bp_df[s] = 0
    k = bp_df[['stay_id', 'cur_bp_time', 'otj_filter', s]].groupby(['stay_id']).apply(
        lambda x: pd.Series(rolling_gp(x, kernel=sigma * RBF(sigma),
                                       alpha=alpha)))
    bp_df['gp'] = k.reset_index(0, drop=True)

    sigma, alpha, l = 2, 0.1, 5
    s = f'gp_sigma-{sigma}_l-{l}_alp-{alpha}'
    bp_df[f'gp_sigma-{sigma}_l-{l}_alp-{alpha}'] = 0
    k = bp_df[['stay_id', 'cur_bp_time', 'otj_filter', 'gp']].groupby(['stay_id']).apply(
        lambda x: pd.Series(rolling_gp(x, kernel=sigma * RBF(sigma),
                                       alpha=alpha)))
    bp_df['gp'] = k.reset_index(0, drop=True)
    return bp_df


def add_rolling_statistics(bp_df: pd.DataFrame, window_size=10):
    bp_df[f'rolling_{window_size}_mean'] = bp_df.groupby("stay_id")["cur_bp"].rolling(window_size,
                                                                                      center=True).mean().reset_index(0,
                                                                                                                      drop=True)
    bp_df[f'rolling_{window_size}_std'] = bp_df.groupby("stay_id")["cur_bp"].rolling(window_size,
                                                                                     center=True).std().reset_index(0,
                                                                                                                    drop=True)
    bp_df[f'rolling_{window_size}_min'] = bp_df.groupby("stay_id")["cur_bp"].rolling(window_size,
                                                                                     center=True).min().reset_index(0,
                                                                                                                    drop=True)
    bp_df[f'rolling_{window_size}_max'] = bp_df.groupby("stay_id")["cur_bp"].rolling(window_size,
                                                                                     center=True).max().reset_index(0,
                                                                                                                    drop=True)
    bp_df[f'rolling_{window_size}_median'] = bp_df.groupby("stay_id")["cur_bp"].rolling(
        window_size, center=True).median().reset_index(0, drop=True)

    bp_df[f'rolling_{window_size}_mean_otj'] = bp_df.groupby("stay_id")["otj_filter"].rolling(window_size,
                                                                                              center=True).mean().reset_index(
        0,
        drop=True)
    bp_df[f'rolling_{window_size}_std_otj'] = bp_df.groupby("stay_id")["otj_filter"].rolling(window_size,
                                                                                             center=True).std().reset_index(
        0,
        drop=True)
    bp_df[f'rolling_{window_size}_min_otj'] = bp_df.groupby("stay_id")["otj_filter"].rolling(window_size,
                                                                                             center=True).min().reset_index(
        0,
        drop=True)
    bp_df[f'rolling_{window_size}_max_otj'] = bp_df.groupby("stay_id")["otj_filter"].rolling(window_size,
                                                                                             center=True).max().reset_index(
        0,
        drop=True)
    bp_df[f'rolling_{window_size}_median_otj'] = bp_df.groupby("stay_id")["otj_filter"].rolling(
        window_size, center=True).median().reset_index(0, drop=True)
    return bp_df


def filter_by_nor(bp_df, break_size=30, min_hrs_to_nor=12, max_hrs_from_admission=48, max_nor=110,
                  min_time_of_stay_entries=6 * 12):
    # filter all entries with drugrate > max_nor
    bp_df = bp_df[bp_df['drugrate'] <= max_nor]

    # filter all entries after 48 hours
    first_measurment_time = bp_df[['stay_id', 'cur_bp_time', 'drugrate']].groupby('stay_id').apply(
        lambda x: x['cur_bp_time'].min())
    bp_df = bp_df.groupby('stay_id').apply(
        lambda x: x[(x['cur_bp_time'] <= first_measurment_time[x.name] + max_hrs_from_admission * 60)]).reset_index(
        level=0, drop=True)

    # filter all patients that didn't recive NOR in first first_nor_time hours and entries after NOR wasn't given
    first_not_time = bp_df.groupby('stay_id').apply(lambda x: x[~x['drugrate'].isna()]['cur_bp_time'].min())
    patients_with_first_nor_under_24 = first_not_time[first_not_time <= min_hrs_to_nor * 60].index
    bp_df = bp_df[bp_df['stay_id'].isin(patients_with_first_nor_under_24)]

    pat_at_stage['received NOR within 24 hours'] = bp_df['stay_id'].nunique()

    bp_df['nor_bigger_than_0'] = bp_df['drugrate'] > 0
    # find the largest cur_bp_time that nor_bigger_than_0 is true
    last_nor_time = bp_df.groupby('stay_id').apply(lambda x: x.loc[x['nor_bigger_than_0'], 'cur_bp_time'].max())

    # filer all rows with cur_bp_time that is larger than last_nor_time
    bp_df = bp_df.groupby('stay_id').apply(lambda x: x[(x['cur_bp_time'] <= last_nor_time[x.name])]).reset_index(
        level=0,
        drop=True)

    # remove all rows after a break larger than break_size
    a = bp_df.groupby('stay_id').apply(lambda x: x['interval'].cummax()).reset_index(level=0, drop=True)
    bp_df['cummax_interval'] = a

    # filter all rows with cummax_interval that is larger than 30
    bp_df = bp_df[bp_df['cummax_interval'] <= break_size]

    no_nor = (bp_df.groupby('stay_id')['drugrate'].transform(lambda x: x.sum() == 0))
    no_nor = no_nor[no_nor].index.tolist()
    bp_df = bp_df[~bp_df['stay_id'].isin(no_nor)]

    # filter all patients with less than 6 hours of stay
    bp_by_stay_id = bp_df.groupby('stay_id').count()
    short_stay = bp_by_stay_id[bp_by_stay_id['cur_bp_time'] < min_time_of_stay_entries].index.tolist()
    bp_df = bp_df[~bp_df['stay_id'].isin(short_stay)]

    pat_at_stage['6 hours of uninterrupted valid MAP'] = bp_df['stay_id'].nunique()

    return bp_df


def load_bp_salz(bp_max=160, bp_min=30):
    d703 = pd.read_csv('data_float_m_703.csv')
    d703 = d703[d703['Offset'] % 300 == 0]
    d703 = d703[d703['Offset'] <= 172800]

    d_cases = pd.read_csv('cases.csv.gz', compression='gzip')
    d_cases['WeightOnAdmission'] /= 1000

    medication = pd.read_csv('medication.csv.gz', compression='gzip')
    vas = medication[medication['DrugID'] == 1550]
    vas_ids = list(vas['CaseID'].unique())
    dop = medication[medication['DrugID'] == 1618]
    dop_ids = list(dop['CaseID'].unique())
    epi = medication[medication['DrugID'] == 1502]
    epi_ids = list(epi['CaseID'].unique())

    other_drugs_ids = list(set(vas_ids + dop_ids + epi_ids))
    d1562 = medication[medication['DrugID'] == 1562]
    d1562 = d1562[d1562['OffsetDrugEnd'] <= 172800]
    d1562 = d1562[~d1562['CaseID'].isin(other_drugs_ids)]

    d1562 = medication[medication['DrugID'] == 1562]
    d1562 = d1562[d1562['OffsetDrugEnd'] <= 172800]

    id_weight_dict = dict(zip(d_cases['CaseID'], d_cases['WeightOnAdmission']))
    pat_at_stage['with NOR'] = d1562['CaseID'].nunique()
    intersection_ids = list(set(d703['CaseID'].unique()).intersection(d1562['CaseID'].unique()))

    # Create a boolean mask to filter rows from d703 and d1562
    mask1 = d703['CaseID'].isin(intersection_ids)
    mask2 = d1562['CaseID'].isin(intersection_ids)

    # Apply the boolean mask to filter rows from d703 and d1562
    df1 = d703.loc[mask1, ['CaseID', 'Offset', 'Val']]
    df2 = d1562.loc[mask2, ['CaseID', 'Offset', 'OffsetDrugEnd', 'AmountPerMinute']]

    # Concatenate df1 and df2
    df3 = pd.concat([df1, df2], axis=0)

    # Concatenate df3 with df4
    df3.reset_index(drop=True, inplace=True)

    df3['Offset'] /= 60
    df3['OffsetDrugEnd'] /= 60
    df3['AmountPerMinute'] *= 1000000

    df3.rename(columns={
        'CaseID': 'stay_id',
        'Val': 'cur_bp',  # Rename 'Val' to 'Value'
        'Offset': 'cur_bp_time',  # Rename 'Drug Rate' to 'Rate'
        'AmountPerMinute': 'drugrate'  # Rename 'Offset' to 'Offset Range'
    }, inplace=True)

    death_dict = {2202: 'ICU survivor', 2212: 'Unknown', 2215: 'ICU non-survivor'}
    death_per_pat_dict = dict(zip(d_cases['CaseID'], d_cases['DischargeState']))

    df3['DischargeState'] = df3['stay_id'].map(death_per_pat_dict)
    df3['DischargeState'] = df3['DischargeState'].map(death_dict)

    df3['weight'] = df3['stay_id'].map(id_weight_dict)

    df3.to_csv('five_minutes_resolution.csv', index=False)

    bp_df = pd.read_csv('five_minutes_resolution.csv')
    pat_at_stage['with only NOR'] =  bp_df.stay_id.nunique()
    bp_df = bp_df.sort_values(['stay_id', 'cur_bp_time'])
    # print(bp_df.drugrate.isna().sum())
    bp_df['drugrate'] = bp_df.groupby('stay_id')['drugrate'].transform(lambda x: x.ffill())
    # print(bp_df.drugrate.isna().sum())
    # print the ampunt of stay_id's that have more than 1 nor measurement and the amount of stay_id's that have 0 nor measurements
    # print(bp_df.groupby('stay_id')['drugrate'].count().value_counts())
    # filter all stay_id's that have 0 nor measurements
    bp_df = bp_df.groupby('stay_id').filter(lambda x: x['drugrate'].count() > 0)
    bp_df = bp_df.groupby('stay_id').filter(lambda x: x['drugrate'].count() > 0)
    bp_df = bp_df[bp_df['cur_bp'] <= bp_max]
    bp_df = bp_df[bp_df['cur_bp'] >= bp_min]
    bp_df = bp_df.loc[~bp_df['cur_bp'].isna(), :]
    # add a column with the time difference between the current and the previous measurement called interval
    bp_df['interval'] = bp_df.groupby('stay_id')['cur_bp_time'].diff()

    return bp_df


global pat_at_stage
if __name__ == "__main__":
    # pat_at_stage = {}
    # big_bp = load_bp(num_of_pats=0)
    # big_bp = filter_by_nor(big_bp, break_size=30)
    #
    # # make bar plot of the amount of patients at each stage with sorted values using plotly
    # pat_at_stage = pd.DataFrame.from_dict(pat_at_stage, orient='index', columns=['Amount of Patients'])
    # pat_at_stage = pat_at_stage.sort_values(by='Amount of Patients', ascending=False)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Bar(x=pat_at_stage.index, y=pat_at_stage['Amount of Patients']))
    # # add value labels on top of the bars and make them bigger
    # fig.update_traces(texttemplate='%{y}', textposition='outside', textfont_size=20)
    # fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')
    #
    # fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    # # fig.update_layout(title_text='Amount of Patients at Each Stage', title_x=0.5)
    # fig.update_layout(xaxis_title='Stage', yaxis_title='Amount of Patients')
    # # make fig size  big enough to show all labels horizontally
    # fig.update_layout(height=1000, width=1500)
    # # make the labels horizontal
    # fig.update_xaxes(tickangle=0)
    # # make tick labels bigger
    # fig.update_layout(xaxis=dict(tickfont=dict(size=20)))
    # fig.update_layout(yaxis=dict(tickfont=dict(size=20)))
    #
    # # make bar width bigger
    # fig.update_traces(marker=dict(line=dict(width=4)))
    #
    # # make axis titles bigger
    # fig.update_layout(xaxis=dict(title=dict(font=dict(size=20))))
    # fig.update_layout(yaxis=dict(title=dict(font=dict(size=20))))
    #
    # fig.show()
    #
    # fig.write_html("../preprocess/patients_at_each_stage.html")
    # fig.write_image("../preprocess/patients_at_each_stage.png")
    #
    # big_bp = remove_one_time_jumps(big_bp)
    #
    # big_bp = smooth_outliers(big_bp, threshold_constant=1.5)
    # for i in range(2, 11):
    #     big_bp = add_rolling_statistics(big_bp, window_size=i)
    #
    # big_bp.to_csv("../preprocess/smooth_bp_eicu2.csv", index=False)

    pat_at_stage = {}

    big_bp_salz = load_bp_salz()
    big_bp_salz = filter_by_nor(big_bp_salz, break_size=30)
    # make bar plot of the amount of patients at each stage with sorted values using plotly
    pat_at_stage = pd.DataFrame.from_dict(pat_at_stage, orient='index', columns=['Amount of Patients'])
    pat_at_stage = pat_at_stage.sort_values(by='Amount of Patients', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pat_at_stage.index, y=pat_at_stage['Amount of Patients']))
    # add value labels on top of the bars and make them bigger
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont_size=20)
    fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    # fig.update_layout(title_text='Amount of Patients at Each Stage', title_x=0.5)
    fig.update_layout(xaxis_title='Stage', yaxis_title='Amount of Patients')
    # make fig size  big enough to show all labels horizontally
    fig.update_layout(height=1000, width=1500)
    # make the labels horizontal
    fig.update_xaxes(tickangle=0)
    # make tick labels bigger
    fig.update_layout(xaxis=dict(tickfont=dict(size=20)))
    fig.update_layout(yaxis=dict(tickfont=dict(size=20)))

    # make bar width bigger
    fig.update_traces(marker=dict(line=dict(width=4)))

    # make axis titles bigger
    fig.update_layout(xaxis=dict(title=dict(font=dict(size=20))))
    fig.update_layout(yaxis=dict(title=dict(font=dict(size=20))))

    fig.show()

    fig.write_html("../preprocess/patients_at_each_stage_zalz.html")
    fig.write_image("../preprocess/patients_at_each_stage_zalz.png")

    big_bp_salz = remove_one_time_jumps(big_bp_salz)
    big_bp_salz = smooth_outliers(big_bp_salz, threshold_constant=1.5)
    for i in range(2, 11):
        big_bp_salz = add_rolling_statistics(big_bp_salz, window_size=i)

    big_bp_salz.to_csv("../preprocess/smooth_bp_salz.csv", index=False)
