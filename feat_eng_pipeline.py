import pandas as pd
import re
import copy
import json

PATH_TO_DATA = 'data/'


def read_data(filename, PATH_TO_DATA):
    df = pd.read_csv(f'{PATH_TO_DATA}{filename}.tsv', sep='\t')
    return df


def preprocess_input_data(df):
    """Splits features into separate columns for further calculations.
    Accepted format includes:
    - id_job - vacancy identifier
    - features columns with numbers depending on the feature position in the concatenated format

    Parameters
    ----------
    df : DataFrame
        processed dataframe with features string column splitted into several columns

    Returns
    -------
    DataFrame
        id_job
            vacancy identifier column
        feature_{feature_code}_{i}
            features retrieved from concatted string
    """
    splitted_features = df['features'].str.split(',', expand=True)
    str_cols = splitted_features.columns[splitted_features.dtypes.eq('object')]
    features_df = pd.concat([
        df.drop(['features'], axis=1),
        splitted_features[str_cols].apply(pd.to_numeric, errors='coerce')
    ], axis=1)

    feature_name = features_df[0].unique()
    # to make sure that the first column contains the name of the features
    assert len(feature_name) == 1
    feature_name = f'feature_{feature_name[0]}'

    features_df.drop(columns=[0], inplace=True)
    features_df.rename(columns={
        col_name: f'{feature_name}_{str(col_name)}'
        for col_name in features_df.columns
        if col_name != 'id_job'
    }, inplace=True)
    return features_df


def set_mean_std_for_features(df, id_col):
    """Calculates and saves mean and standard deviation values for all for provided features columns.

    Parameters
    ----------
    df : DataFrame
        processed dataframe with features string column splitted into several columns
    id_col: string
        the name of the column that stores vacancies identifiers
    """
    global MEAN_FEATURES_VALS
    global STD_FEATURES_VALS
    MEAN_FEATURES_VALS = df.drop(id_col, axis=1).mean().to_dict()
    STD_FEATURES_VALS = df.drop(id_col, axis=1).std().to_dict()


def calculate_features(df, id_col, recalculate_stats=True):
    """Implements standartization of features values and fetches deviation and max index features.

    Parameters
    ----------
    df : DataFrame
        processed dataframe with features string column splitted into several columns
    id_col: string
        the name of the column that stores vacancies identifiers

    Returns
    -------
    DataFrame
        calculated features
    """
    if recalculate_stats:
        set_mean_std_for_features(df, id_col)
    df_features = df[[id_col]].copy()
    get_standardized_feature_col_name.feature_code = ''
    for col in df.drop(id_col, axis=1).columns:
        stand_col_name = get_standardized_feature_col_name(
            col, get_standardized_feature_col_name.feature_code)
        df_features[stand_col_name] = standardize_with_z_score(df[col], col)
        df_features = df_features.merge(get_abs_max_min_deviation(df, id_col))

    return df_features


def standardize_with_z_score(x, feature_name):
    """Applies z-score standartization to feature values using standard devidation and mean
    calculated for training dataset.
    """
    return (x - MEAN_FEATURES_VALS.get(feature_name)) / \
        STD_FEATURES_VALS.get(feature_name)


def get_standardized_feature_col_name(col, feature_code):
    if not get_standardized_feature_col_name.feature_code:
        try:
            get_standardized_feature_col_name.feature_code = re.search(
                'feature_[0-9]+', col).group(0)
        except AttributeError as error:
            print('Feature name does not correspond to the pattern')
    i = get_feature_number(col)
    return f'{get_standardized_feature_col_name.feature_code}_stand_{i}'


def get_feature_number(feature_name):
    """Detects number of feature i if feature column name corresponds to format
    feature_{feature_code}_{feature_number} then the number of feature is fetched.
    """
    feature_code_pattern = re.compile("[0-9]+")
    try:
        i = feature_code_pattern.findall(feature_name)[-1]
    except IndexError as error:
        i = ''
        print('No feature number was found in the gived column name.')
    return i


def get_abs_max_min_deviation(df_features, id_col):
    """Calculates max_feature_2_index and max_feature_2_abs_mean_diff features by transposing
    passed dataframe with processed raw featurs.

    Parameters
    ----------
    df : DataFrame
        processed dataframe with features string column splitted into several columns
    id_col: string
        the name of the column that stores vacancies identifiers

    Returns
    -------
    DataFrame
        with calculated features and vacancy identifier as a separete column
    """
    df_features_transposed = df_features.T
    max_val_col_name = f'max_{get_standardized_feature_col_name.feature_code}_value'
    df_features_transposed.loc[max_val_col_name] = df_features_transposed[1:].max(
    )
    max_index_col_name = f'max_{get_standardized_feature_col_name.feature_code}_index'
    df_features_transposed.loc[max_index_col_name] = df_features_transposed[1:-1].idxmax()
    df_features_transposed = df_features_transposed.loc[[id_col, max_index_col_name, max_val_col_name]].T.merge(
        pd.DataFrame(MEAN_FEATURES_VALS.items(), columns=[max_index_col_name, 'mean_feature_val']))
    abs_mean_diff_col_name = f'max_{get_standardized_feature_col_name.feature_code}_abs_mean_diff'
    df_features_transposed[abs_mean_diff_col_name] = df_features_transposed[max_val_col_name] - \
        df_features_transposed['mean_feature_val']
    df_features_transposed[abs_mean_diff_col_name] = df_features_transposed[abs_mean_diff_col_name].astype(
        float)

    # transforming features names into number of feature i
    df_features_transposed[max_index_col_name] = [get_feature_number(
        x) for x in df_features_transposed[max_index_col_name]]
    df_features_transposed[max_index_col_name] = df_features_transposed[max_index_col_name].astype(
        int)
    return df_features_transposed[[
        id_col, max_index_col_name, abs_mean_diff_col_name]]
