import pandas as pd
import pytest
import json
from feat_eng_pipeline import PATH_TO_DATA, read_data, preprocess_input_data, calculate_features


def test_running_pipeline():

    def run_pipeline(filename, recalculate_stats):
        raw_data = read_data(filename, PATH_TO_DATA)
        raw_data = preprocess_input_data(raw_data)
        features = calculate_features(raw_data, 'id_job', recalculate_stats)
        return features

    train_features = run_pipeline('train', True)
    assert not train_features.empty
    with open('mean_features_values.json') as file:
        train_mean_features_vals_dict = json.load(file)
    with open('std_features_values.json') as file:
        train_std_features_vals_dict = json.load(file)

    test_features = run_pipeline('test', False)
    assert not test_features.empty
    with open('mean_features_values.json') as file:
        test_mean_features_vals_dict = json.load(file)
    with open('std_features_values.json') as file:
        test_std_features_vals_dict = json.load(file)
    # check that stats were not recalcuted
    assert train_mean_features_vals_dict == test_mean_features_vals_dict
    assert train_std_features_vals_dict == test_std_features_vals_dict

    test_features.to_csv(f'{PATH_TO_DATA}test_proc.tsv')
