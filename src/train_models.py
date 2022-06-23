import pandas as pd
import numpy as np
import json
import argparse

# have to run pip install -e . before running
from src.utils import setup_folders, return_static_features, return_dynamic_features
from src.features.build_features import build_features, build_features_LSTM, build_features_seq
from src.models.deterministic.models import RF, LR, LGB, Persistence, my_LSTM

import warnings
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

parser = argparse.ArgumentParser()
parser.add_argument('--tuned_run', type = bool, default = False)
args = parser.parse_args()

# We set the models that we want to loop over
#plants = ["Nazeerabad", "HPP1", "HPP2", "HPP3"]
plants = ["Nazeerabad"]
techs = ["wind", "solar", "agr"]

for plant in plants:
    print(plant)
    for cnt, tech in enumerate(techs):
        print(tech)

        ### We load the data
        if plant == "Nazeerabad":
            path = "data/processed/Nazeerabad/Nazeerabad_OBS_METEO_30min_precleaned.csv"
            df = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])
        elif plant == "HPP1":
            path = "data/processed/HPP1/HPP1_OBS_METEO_30min.csv"
            df = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])
        elif plant == "HPP2":
            path = "data/processed/HPP2/HPP2_OBS_METEO_30min.csv"
            df = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])
        elif plant == "HPP3":
            path = "data/processed/HPP3/HPP3_OBS_METEO_30min.csv"
            df = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])

        ### We specify the features and targets
        features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)

        ### LR
        static = False
        LR_model = LR(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        LR_model.train(plant)

        ### RF
        RF_model = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        RF_model.train(plant)

        ### LGB
        LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        LGB_model.train(plant)

        ### We make static features which contain no power information
        features, targets = return_static_features(plant, tech)
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)

        ### LR
        static = True
        LR_model = LR(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        LR_model.train(plant)

        ### RF
        RF_model = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        RF_model.train(plant)

        ### LGB HP
        f = open(f'src/models/deterministic/params/params_LGB_{plant}.json')
        params = json.load(f)
        f.close()

        ### LGB
        if args.tuned_run == True:
            LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static, **params[tech])
        else:
            LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        LGB_model.train(plant)

        ## LSTM HP
        f = open(f'src/models/deterministic/params/params_LSTM_{plant}.json')
        params = json.load(f)
        f.close()

        ### Create recursive LSTM features
        n_lag = 4; n_out = 1
        dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
        if args.tuned_run == True:
            LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = True, **params[tech])
        else:
            LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = True)
        LSTM_model.train(plant, epochs = params[tech]['epochs'], batch_size = params[tech]['batch_size'])

        ### Create full sequence LSTM features
        n_lag = 144; n_out = 48
        dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
        if args.tuned_run == True:
            LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = False, **params[tech])
        else:
            LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = False)
        LSTM_model.train(plant, epochs = params[tech]['epochs'], batch_size = params[tech]['batch_size'])

        # RF HP
        f = open(f'src/models/deterministic/params/params_RF_{plant}.json')
        params = json.load(f)
        f.close()

        # Create sequential features for RF
        n_lag = 144; n_out = 48
        features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
        dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)

        # RF sequential
        if args.tuned_run == True:
            RF_model_seq = RF(dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq, static = False, **params[tech])
        else:
            RF_model_seq = RF(dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq, static = False)
        RF_model_seq.train(plant)