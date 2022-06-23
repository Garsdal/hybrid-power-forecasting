from re import L
import numpy as np
import pandas as pd
from pyparsing import java_style_comment
import sklearn.ensemble
from sklearn.linear_model import LinearRegression
import pickle
import datetime
from datetime import datetime as dt
from src.utils import setup_folders, shift1, init_logging, LoggingCallback
from src.models.predict_functions import recursive_predict_test_days, predict_full_sequence_days, recursive_predict_test_days_LSTM
import lightgbm as lgb
import os
import joblib

# LSTM
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Masking, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

import logging
import logging.config

class RF:
    def __init__(self, dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, seq = None, runtime = None, wandb = None, **params):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_val = Y_val
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.dt_val = dt_val
        self.features = features
        self.targets = targets
        self.static = static
        self.seq = seq
        self.model_name = "RF"
        self.params = params
        self.wandb = wandb
        self.runtime = runtime
        self.model = None
    
    def train(self, plant):
        # We create a folder
        root_location=os.path.abspath(os.sep)
        scratch_location='work3/s174440'
   
        # We create a specific folder for models when we are tuning
        static_bool = 'static' if self.static else 'dynamic'
        if self.wandb is not None:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}'
            save_folder = f'{root_location}/{scratch_location}/models_tuning/{self.model_name}/{plant}'
            setup_folders(save_folder)
        elif len(self.params) != 0:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models_tuned/{self.model_name}/{plant}'
            setup_folders(save_folder)
        else:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models/{self.model_name}/{plant}'
            setup_folders(save_folder)

        # We create a logger
        logging.config.fileConfig("logger.ini",
                              disable_existing_loggers=True,
                              defaults={'logfilename': f'{save_folder}/logs_{filename_log}.txt'})

        init_logging(logging, self.model_name, plant, self.features, self.targets, self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val, self.params, static_bool)

        # We log each parameter if we have a wandb init
        if self.wandb is not None:
            for key in self.params.keys():
                self.wandb.log({key: self.params[key]})

        # We train the model depending on params available
        if len(self.params) != 0:
            ForestReg = sklearn.ensemble.RandomForestRegressor(random_state = 1, **self.params)
            ForestReg.fit(self.X_train, self.Y_train)
        else:
            ForestReg = sklearn.ensemble.RandomForestRegressor(random_state = 1)
            ForestReg.fit(self.X_train, self.Y_train)

        # We log the feature importance
        logging.info(f"Feature importance: {ForestReg.feature_importances_}")
        
        # We save the model
        self.save_model(ForestReg, plant, save_folder)

    def test(self, filename, set = "test"):
        if self.model is None:
            print("No self.model was found.")
            # We load a model
            model = self.load_model(filename)
        else:
            model = self.model

        if set == "test":
            X = self.X_test
            dt = self.dt_test
        elif set == "val":
            X = self.X_val
            dt = self.dt_val
        
        if self.static:
            Y_pred = model.predict(X)
            return(Y_pred)
        elif self.seq:
            Y_pred = predict_full_sequence_days(model, X, dt)
            return(Y_pred)
        else:
            Y_pred = recursive_predict_test_days(model, X, dt, self.features, self.targets)
            return(Y_pred)

    def save_model(self, model, plant, save_folder):
        static_bool = 'static' if self.static else 'dynamic'
        # When we are tuning models we add a timestamp to keep them unique
        if self.wandb is not None:
            filename = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}.joblib'
        else:
            filename = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}.joblib'

        filepath = "/".join([save_folder, filename])
        joblib.dump(model, filepath, compress=3)
        logging.info(f'Model saved to: {filepath}')

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print("Model loaded from:", filename)
        return(self.model)


class LGB:
    def __init__(self, dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, runtime = None, wandb = None, **params):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_test = Y_test  
        self.Y_val = Y_val  
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.dt_val = dt_val
        self.features = features
        self.targets = targets
        self.static = static
        self.model_name = "LGB"
        self.wandb = wandb
        self.params = params
        self.runtime = runtime
        self.model = None
    
    def train(self, plant):
        # We create a folder
        root_location=os.path.abspath(os.sep)
        scratch_location='work3/s174440'

        # We create a specific folder for models when we are tuning
        static_bool = 'static' if self.static else 'dynamic'
        if self.wandb is not None:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}'
            save_folder = f'{root_location}/{scratch_location}/models_tuning/{self.model_name}/{plant}'
            setup_folders(save_folder)
        elif len(self.params) != 0:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models_tuned/{self.model_name}/{plant}'
            setup_folders(save_folder)
        else:
            filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models/{self.model_name}/{plant}'
            setup_folders(save_folder)

        # We create a logger
        logging.config.fileConfig("logger.ini",
                              disable_existing_loggers=True,
                              defaults={'logfilename': f'{save_folder}/logs_{filename_log}.txt'})

        init_logging(logging, self.model_name, plant, self.features, self.targets, self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val, self.params, static_bool)

        # We log each parameter if we have a wandb init
        if self.wandb is not None:
            for key in self.params.keys():
                self.wandb.log({key: self.params[key]})

        # We train the model depending on params available
        if len(self.params) != 0:
            LGB = lgb.LGBMRegressor(**self.params)
            LGB.fit(X = self.X_train, y = self.Y_train)
        else:
            LGB = lgb.LGBMRegressor()
            LGB.fit(X = self.X_train, y = self.Y_train)

        # We save the model
        self.save_model(LGB, plant, save_folder)

    def test(self, filename, set = "test"):
        if self.model is None:
            print("No self.model was found.")
            # We load a model
            model = self.load_model(filename)
        else:
            model = self.model

        if set == "test":
            X = self.X_test
            dt = self.dt_test
        elif set == "val":
            X = self.X_val
            dt = self.dt_val
        
        if self.static:
            Y_pred = model.predict(X)
            return(Y_pred)
        else:
            Y_pred = recursive_predict_test_days(model, X, dt, self.features, self.targets)
            return(Y_pred)

    def save_model(self, model, plant, save_folder):
        # When we are tuning models we add a timestamp to keep them unique
        static_bool = 'static' if self.static else 'dynamic'
        if self.wandb is not None:
            filename = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}.sav'
        else:
            filename = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}.sav'

        filepath = "/".join([save_folder, filename])
        pickle.dump(model, open(filepath, 'wb'))
        logging.info(f'Model saved to: {filepath}')

    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))
        print("Model loaded from:", filename)
        return(self.model)

# Doesn't have a validation set because we don't hyperparameter tune
class LR:
    def __init__(self, dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, **params):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_val = Y_val
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.dt_val = dt_val
        self.features = features
        self.targets = targets
        self.static = static
        self.model_name = "LR"
        self.params = params
    
    def train(self, plant):
        # We set up a log for training
        static_bool = 'static' if self.static else 'dynamic'
        filename_log = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}'

        # We create a folder
        root_location=os.path.abspath(os.sep)
        scratch_location='work3/s174440'
        save_folder = f'{root_location}/{scratch_location}/models/{self.model_name}/{plant}'
        setup_folders(save_folder)

        # We create a logger
        logging.config.fileConfig("logger.ini",
                              disable_existing_loggers=True,
                              defaults={'logfilename': f'{save_folder}/logs_{filename_log}.txt'})

        init_logging(logging, self.model_name, plant, self.features, self.targets, self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val, self.params, static_bool)

        LinReg = LinearRegression() 
        LinReg.fit(self.X_train, self.Y_train)

        # We save the model
        self.save_model(LinReg, plant, save_folder)

    def test(self, filename):
        # We load a model
        model = self.load_model(filename)

        if self.static:
            Y_pred = model.predict(self.X_test)
            return(Y_pred)
        else:
            Y_pred = recursive_predict_test_days(model, self.X_test, self.dt_test, self.features, self.targets)
            return(Y_pred)

    def save_model(self, model, plant, save_folder):
        static_bool = 'static' if self.static else 'dynamic'
        filename = f'{self.model_name}_{plant}_{static_bool}_numf{len(self.features)}_{self.targets[0]}.sav'

        filepath = "/".join([save_folder, filename])
        pickle.dump(model, open(filepath, 'wb'))
        logging.info(f'Model saved to: {filepath}')

    def load_model(self, filename):
        model = pickle.load(open(filename, 'rb'))
        print("Model loaded from:", filename)
        return(model)

# Doesn't have a validation set because we don't hyperparameter tune
class Persistence:
    def __init__(self, dt_train, dt_test, X_train, X_test, Y_train, Y_test, tech):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.tech = tech

    def test(self):
        # We produce a 'naive' forecast for DAH based on the previous value from 23:30

        # We get midnights from all test days
        test_days = self.dt_test.strftime('%Y-%m-%d').unique()
        midnights = pd.to_datetime(test_days).strftime('%Y-%m-%d %H:%M:%S')
        midnights_prev = pd.to_datetime(midnights) + datetime.timedelta(days=0,minutes=-30)

        # We grab all the days where we have a index in dt_train the previous night | such that we can get a value
        forecast_dt = midnights_prev.intersection(self.dt_train)

        # We get all indices 
        forecast_idx = [list(self.dt_train).index(x) for x in forecast_dt]

        # For solar persistence we have to pad the values before we look back
        df_index = pd.DataFrame(index = pd.date_range(self.dt_train[0], self.dt_train[-1], freq = "30min"))
        df_merged = df_index.join(pd.DataFrame(self.Y_train, index = self.dt_train))
        #Y_train_padded = np.array(df_merged.values)

        # We create the naive forecast for all forecasts days
        dt_list = []
        val_list = []
        for cnt, day in enumerate(forecast_dt):
            # We create the timeindex for the predictions
            dt_start = day + datetime.timedelta(days=0,minutes=30) # we go from 23:30 to 00:00
            dt_end = dt_start + datetime.timedelta(days=1,minutes=-30) # we go from 00:00 to 23:30
            dt_range = pd.date_range(dt_start, dt_end, freq = "30min")
            dt_list.append(dt_range) # we add the dt range
            
            # we grab a datetime for the previous day
            dt_previous_day = (day + datetime.timedelta(days=0,minutes=-30)).strftime('%Y-%m-%d')

            if self.tech in ['solar']:
                # We grab the previous days values
                df_solar_persistence = df_merged.loc[dt_previous_day]

                # We pad the previous days values
                df_solar_persistence_padded = pd.DataFrame(index = dt_range).join(df_solar_persistence)
                
                # We append the values
                val_list.append(np.concatenate(np.array(df_solar_persistence.values)))
            elif self.tech in ["wind", "agr"]:
                # We create the naive forecast from Y_train with the correct index
                val_list.append(np.repeat(self.Y_train[forecast_idx[cnt]], 48))
            
        dt_list = np.concatenate(np.array(dt_list))
        val_list = np.concatenate(np.array(val_list))

        # For now we match to the test set but maybe not if we end up padding with nans
        df_persistence = pd.DataFrame({'power_pred': val_list}, index = dt_list)

        # For now we join the df_persistence on the dt_test | we are missing values for some days
        df_index = pd.DataFrame(index = self.dt_test)
        df_merged = df_index.join(df_persistence)
        
        # We get the predictions
        Y_pred = np.concatenate(df_merged.values)

        return(Y_pred)

class my_LSTM:
    def __init__(self, dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, recursive = False, runtime = None, wandb = None, **params):
        # LSTM problems [samples, timesteps, features]
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_val = Y_val
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.dt_val = dt_val
        self.features = features
        self.targets = targets
        self.recursive = recursive
        self.model_name = "LSTM"
        self.params = params
        self.wandb = wandb
        self.runtime = runtime
        self.model = None
        
    def train(self, plant, epochs = 5, batch_size = 1000, neurons = [100,50,25]):
        # We grab n_lag and n_out for filenames
        n_lag = self.X_test.shape[1]
        n_out = self.Y_test.shape[1]

        # We locate the scratch workspace
        root_location=os.path.abspath(os.sep)
        scratch_location='work3/s174440'

        # We create a specific folder for models when we are tuning
        if self.wandb is not None:
            filename_log = f'{self.model_name}_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}'
            save_folder = f'{root_location}/{scratch_location}/models_tuning/{self.model_name}/{plant}'
            setup_folders(save_folder)
        elif len(self.params) != 0:
            filename_log = f'{self.model_name}_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models_tuned/{self.model_name}/{plant}'
            setup_folders(save_folder)
        else:
            filename_log = f'{self.model_name}_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(self.features)}_{self.targets[0]}'
            save_folder = f'{root_location}/{scratch_location}/models/{self.model_name}/{plant}'
            setup_folders(save_folder)

        # We create a logger
        logging.config.fileConfig("logger.ini",
                              disable_existing_loggers=True,
                              defaults={'logfilename': f'{save_folder}/logs_{filename_log}.txt'})

        init_logging(logging, self.model_name, plant, self.features, self.targets, self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val, self.params, recursive_bool = self.recursive)

        # We log each parameter if we have a wandb init
        if self.wandb is not None:
            for key in self.params.keys():
                self.wandb.log({key: self.params[key]})

        #Number of neurons per layer | we only use the dict if it contains keys
        if len(self.params) != 0:
            i1 = self.params['neurons_l1']
            i2 = self.params['neurons_l2']
            i3 = self.params['neurons_l3']
            lr = self.params['lr']
            act_fnc = self.params['activation']
        else:
            i1 = neurons[0]
            i2 = neurons[1]
            i3 = neurons[2]
            lr = 0.01
            act_fnc = 'tanh'

        no_features = int(self.X_train.shape[2])
        pad_value = 999

        # for multiple model creation - clear  the previous DAG
        K.clear_session() 

        ### create model
        self.model = Sequential()

        # Masking layer (for the pad_value) - to tell NN not the update the weights for NaN inputs
        self.model.add(Masking(mask_value=pad_value, input_shape=(None, no_features)))

        # First LSTM layer
        self.model.add(LSTM(i1, 
                    return_sequences=True,  # important to add it to ensure the following LSTM layers will have the same input shape
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),                
                    kernel_initializer='glorot_uniform', # seems to be the best for tanh
                    bias_initializer='zeros',
                    activation=act_fnc))

        # Second LSTM layer
        self.model.add(LSTM(i2, 
                    return_sequences=True,
                    activation=act_fnc))

        # Third LSTM layer
        self.model.add(LSTM(i3, 
        #                return_sequences=True,
                    activation=act_fnc
                    ))

        # Output Layer
        self.model.add(Dense(self.Y_train.shape[1]
        #                 activation='tanh'
                        ))

        self.model.summary(print_fn=logging.info)

        # compile the model
        opt = optimizers.Adam(learning_rate = lr)
        self.model.compile(loss='mean_squared_error', optimizer=opt)

        # fit the model and store the graphs and performance to be used in TensorBoard (optional)
        model_params = np.sum([K.count_params(w) for w in self.model.trainable_weights])

        history = self.model.fit(self.X_train, self.Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(self.X_test, self.Y_test),
                        shuffle = True,
                        callbacks=[LoggingCallback(logging.info)])
                   #callbacks=[tbGraph])

        # We save the model
        self.save_model(self.model, plant, save_folder, epochs, n_lag, n_out)

        return(self.model)

    def test(self, filename, set = "test"):
        if self.model is None:
            print("No self.model was found.")
            # We load a model
            model = self.load_model(filename)
        else:
            model = self.model

        if set == "test":
            X = self.X_test
            dt = self.dt_test
        elif set == "val":
            X = self.X_val
            dt = self.dt_val

        # Recursive implies full step-ahead sequence prediction (n_out = 1)
        if self.recursive:
            # The column to update recursively is the target column
            col_idx = np.where(self.targets[0] == np.array(self.features))[0][0]
            print("Features:", self.features)
            print("Updating recursively column index:", col_idx)
            Y_pred = recursive_predict_test_days_LSTM(model, X, dt, col_idx)
            return(Y_pred)
        # Static implies full day-ahead sequence prediction (n_out = 48)
        else:
            Y_pred = predict_full_sequence_days(model, X, dt)
            return(Y_pred)

    def save_model(self, model, plant, save_folder, epochs, n_lag, n_out):
        # When we are tuning models we add a timestamp to keep them unique
        if self.wandb is not None:
            filename = f'{self.model_name}_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(self.features)}_{self.targets[0]}_{self.runtime}.h5'
        else:
            filename = f'{self.model_name}_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(self.features)}_{self.targets[0]}.h5'

        filepath = "/".join([save_folder, filename])
        model.save(filepath)
        logging.info(f'Model saved to: {filepath}')

    def load_model(self, filename):
        self.model = load_model(filename)
        print("Model loaded from:", filename)
        self.model.summary()
        return(self.model)