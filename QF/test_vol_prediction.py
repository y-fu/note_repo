# Tests
import numpy as np
import pandas as pd
import pytest
from vol_predict import VolPredictor

predictor = VolPredictor()

def test_missing_infinite_data():
    X_df = predictor.calculate_return_and_vol(data_df=predictor.get_market_data(ticker='GLD', retrieve_period='1mo'), window_size=10)

    assert not X_df.isnull().values.any(), 'There is NaN value in X'

def test_data_restructure_without_targets():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    steps = 2
    sequence_length = 3

    expected_length = 6
    expected_last_sample = [6, 8, 10]

    X = predictor.restruncture_data(data, vol_window_days=steps, sequence_months=sequence_length)
    
    assert len(X) == expected_length
    assert (X[-1] == expected_last_sample).all()
    assert len(X[0]) == sequence_length

def test_data_restructure_with_targets():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    steps = 2
    sequence_length = 3

    expected_length = 4
    expected_last_sample = [4, 6, 8]
    expected_last_target = [10]

    X, y = predictor.restruncture_data(data=data, vol_window_days=steps, sequence_months=sequence_length, has_target=True)

    assert len(X) == len(y)
    assert len(X) == expected_length, f'expect {expected_length} samples but has {len(X)} : {X}'
    assert (X[-1] == expected_last_sample).all(), f'actural: {X}'
    assert y[-1] == expected_last_target, f'actural: {y}'
    assert len(X[0]) == sequence_length

def test_data_normalise_range():
    sequence_length = 3
    col_name = 'Rv_3'
    data = pd.DataFrame(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=[col_name])
    scaled_data = predictor.normalise_data(data, sequence_length)

    assert max(scaled_data['Scaled_Rv_3']) <= 1
    assert max(scaled_data['Scaled_Rv_3']) >= 0

    
# test_missing_infinite_data()
# test_data_restructure_without_targets()
# test_data_restructure_with_targets()
# test_data_normalise_range()