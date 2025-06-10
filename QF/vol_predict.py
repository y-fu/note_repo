import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
from sklearn import preprocessing
# from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit
from pandas.tseries.offsets import BDay
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization


class VolPredictor:

    def __init__(self, vol_col_prefix='Rv', feature_col_prefix='Scaled'):
        self._VCOL_PREFIX = vol_col_prefix
        self._FEATURE_PREFIX = feature_col_prefix
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    def get_asset(self, ticker):
        return yf.Ticker(ticker)
    
    def get_market_data(self, ticker, retrieve_period):
         asset = self.get_asset(ticker)
         return asset.history(period=retrieve_period).reset_index()
    
    def calculate_return_and_vol(self, data_df, window_size=21):
        data_df['Date'] = data_df['Date'].apply(lambda dt: dt.date())
        data_df['Log_Returns'] = np.log(data_df['Close'] / data_df['Close'].shift(1))

        col_name = f'{self._VCOL_PREFIX}_{window_size}'
        data_df[col_name] = data_df['Log_Returns'].rolling(window=window_size).std() * np.sqrt(252) # annulise daily volatility
        
        return data_df.iloc[window_size:]
    
    def normalise_data(self, data_df, cols, scaler=None):
        if scaler:
            self.scaler = scaler
        else:
            self.scaler.fit(data_df[cols].values)
        
        names = []
        for col in cols:
            names.append(f'{self._FEATURE_PREFIX}_{col}')
        
        data_df[names] = self.scaler.transform(data_df[cols].values)

        return data_df

    def restruncture_data(self, data, vol_window_days, sequence_months, target_steps=0):
        sequence_length = sequence_months
        steps = vol_window_days
        X = []
        
        num_samples = steps * (sequence_length - 1 + target_steps) + 1
        max_start = len(data) - num_samples + 1

        if target_steps:
            y = []
            for i in range(max_start):
                X_indices = [i + j * steps for j in range(sequence_length)]
                y_indices = i + steps * (sequence_length + target_steps - 1)
                X.append(data[X_indices])
                y.append(data[y_indices])
            return np.array(X), np.array(y)
        else:
            for i in range(max_start):
                X_indices = [i + j*steps for j in range(sequence_length)]
                X.append(data[X_indices])
            return np.array(X)

    def smape(self, y_true, y_pred):
            epsilon = 1e-7
            numerator = tf.abs(y_pred-y_true)
            denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2 + epsilon
            return tf.reduce_mean(numerator / denominator) * 100
    
    def build_model(self, hp, sequence_length_months=6, num_features=1, num_outputs=1):  
        model = Sequential()
        model.add(Input(shape=(sequence_length_months, num_features)))
        
        num_layers = hp.Int("number of hidden layers", 1, 3)
        for i in range(num_layers):
            # LSTM with return_sequences=True to pass sequence to next layer
            model.add(LSTM(units=hp.Int(f'units_{i+1}', min_value=32, max_value=128, step=32), 
                           return_sequences=(i < num_layers - 1))
                           )
            model.add(Dropout(hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.3, step=0.1)))

        # Output layer
        model.add(Dense(num_outputs, activation='linear')) 

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')), 
                      loss='mse',
                      metrics=[self.smape]) 
        return model
    
    def train_model(self, X, y, max_trials=20, model_path=None):
        log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        sequence_length = X.shape[1]
        num_features = X.shape[2]
        num_outputs = y.shape[1]

        tuner = BayesianOptimization(hypermodel=lambda hp: self.build_model(hp, sequence_length_months=sequence_length, num_features=num_features, num_outputs=num_outputs),
                                     objective=['val_loss'],
                                     max_trials=max_trials,
                                     num_initial_points=5,
                                     directory=log_dir,
                                     project_name='vol_pred',
                                     overwrite=True
                                     )
        
        tb_callback = TensorBoard(log_dir=log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # tscv = TimeSeriesSplit(n_splits=6)
        ss = ShuffleSplit(n_splits=6, random_state=42, test_size=0.25)
        
        # for fold, (train_idx, val_idx)in enumerate(tscv.split(X)):
        for train_idx, val_idx in ss.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]

            X_val = X[val_idx]
            y_val = y[val_idx]

            tuner.search(X_train, y_train, 
                         validation_data=(X_val, y_val), 
                         epochs=50,
                         callbacks=[tb_callback, keras.callbacks.EarlyStopping(patience=5)]
                        )
        if model_path:
            tuner.get_best_models(1)[0].save(model_path, save_format='keras')
            
        return tuner
    
    def load_model(self, model_path='./model.keras'):
        return keras.models.load_model(model_path, 
                                       custom_objects={"smape": self.smape})

    def generate_model_performance(self, tuner, X_test, y_test, X_all, y_all):
        trials = tuner.oracle.trials
        models = tuner.get_best_models(num_models=len(trials))

        trial_data = []

        for trial_id, trial in tuner.oracle.trials.items():
            trial_id = int(trial_id)

            data = {
                "trial_id":trial_id,
                "score":trial.score
            }
            test_loss, test_metric = models[trial_id].evaluate(X_test, y_test)
            all_loss, all_metric = models[trial_id].evaluate(X_all, y_all)

            data.update({'test_loss': round(test_loss, 4), 
                        'test_metric': round(test_metric, 4),
                        'all_loss': round(all_loss, 4),
                        'all_metric': round(all_metric, 4)})

            data.update(trial.hyperparameters.values)

            for metric in trial.metrics.metrics.keys():
                data[f'{metric}_last'] = trial.metrics.get_last_value(metric)

            trial_data.append(data)

            trial_data_df = pd.DataFrame(trial_data)
        
        return trial_data_df


