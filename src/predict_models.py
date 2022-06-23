import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import argparse
import os

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
plants = ["Nazeerabad", "HPP1", "HPP2", "HPP3"]
techs = ["wind", "solar", "agr"]

# For scratch model loading
root_location=os.path.abspath(os.sep)
scratch_location='work3/s174440'

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

        ### Persistence
        PST_model = Persistence(dt_train, dt_test, X_train, X_test, Y_train, Y_test, tech)
        Y_pred_PST = PST_model.test()

        ### LR
        static = False
        static_bool = 'static' if static else 'dynamic'
        LR_model = LR(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        Y_pred_LR = LR_model.test(f"{root_location}/{scratch_location}/models/LR/{plant}/LR_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.sav")

        ### RF
        RF_model = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        Y_pred_RF = RF_model.test(filename = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.joblib")

        ### LGB
        LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        Y_pred_LGB = LGB_model.test(filename = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.sav")

        ### We make static features which contain no power information
        features, targets = return_static_features(plant, tech)
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)

        ### LR
        static = True
        static_bool = 'static' if static else 'dynamic'
        LR_model = LR(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        Y_pred_LR_static = LR_model.test(f"{root_location}/{scratch_location}/models/LR/{plant}/LR_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.sav")

        ### RF
        RF_model = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        Y_pred_RF_static = RF_model.test(filename = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.joblib")

        ### LGB
        LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)
        if args.tuned_run == True:
            Y_pred_LGB_static = LGB_model.test(filename = f"{root_location}/{scratch_location}/models_tuned/LGB/{plant}/LGB_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.sav")
        else:
            Y_pred_LGB_static = LGB_model.test(filename = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_{static_bool}_numf{len(features)}_{targets[0]}.sav")

        ### Create recursive LSTM features
        features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
        n_lag = 4; n_out = 1
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)

        ### Recursive LSTM model + predictions
        epochs = 20
        LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = True)

        if args.tuned_run == True:
            Y_pred_LSTM_recursive = LSTM_model.test(filename = f"{root_location}/{scratch_location}/models_tuned/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(features_LSTM)}_{targets_LSTM[0]}.h5")
        else:
            Y_pred_LSTM_recursive = LSTM_model.test(filename = f"{root_location}/{scratch_location}/models/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(features_LSTM)}_{targets_LSTM[0]}.h5")
        Y_pred_LSTM_recursive_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_LSTM_recursive, index = dt_test_LSTM)).values)

        ### Create full sequence LSTM features
        features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
        n_lag = 144; n_out = 48
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)

        ### Full sequence LSTM
        epochs = 20
        LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = False)
    
        if args.tuned_run == True:
            Y_pred_LSTM_seq = LSTM_model.test(filename = f"{root_location}/{scratch_location}/models_tuned/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(features_LSTM)}_{targets_LSTM[0]}.h5")
        else:
            Y_pred_LSTM_seq = LSTM_model.test(filename = f"{root_location}/{scratch_location}/models/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(features_LSTM)}_{targets_LSTM[0]}.h5")
        Y_pred_LSTM_seq_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_LSTM_seq, index = dt_test_LSTM)).values)

        # Create fullday features for RF
        n_lag = 144; n_out = 48
        features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)

        ### RF (full sequence) | has extra arguments for padding
        static, seq = False, True
        static_bool = 'static' if static else 'dynamic'
        RF_model_seq = RF(dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq, static = static, seq = seq)
        
        if args.tuned_run == True:
            Y_pred_RF_seq = RF_model_seq.test(filename = f"{root_location}/{scratch_location}/models_tuned/RF/{plant}/RF_{plant}_{static_bool}_numf{len(features_seq)}_{targets_seq[0]}.joblib")
        else:
            Y_pred_RF_seq = RF_model_seq.test(filename = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_{static_bool}_numf{len(features_seq)}_{targets_seq[0]}.joblib")
        Y_pred_RF_seq_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_RF_seq, index = dt_test_seq)).values)

        ### Gather the results
        df_preds = pd.DataFrame({'Y_pred_RF(dynamic)': Y_pred_RF, 
                                'Y_pred_LR(dynamic)': Y_pred_LR, 
                                'Y_pred_RF(static)': Y_pred_RF_static, 
                                'Y_pred_LR(static)': Y_pred_LR_static, 
                                'Y_pred_PST': Y_pred_PST, 
                                'Y_pred_LSTM(recursive)': Y_pred_LSTM_recursive_padded, 
                                'Y_pred_LSTM(full day)': Y_pred_LSTM_seq_padded, 
                                'Y_pred_LGB(dynamic)': Y_pred_LGB,
                                'Y_pred_LGB(static)': Y_pred_LGB_static,
                                'Y_pred_RF(full day)': Y_pred_RF_seq_padded,
                                'Y_true': Y_test}, 
                                index = dt_test)

        # We pad the results
        df_index = pd.date_range(df_preds.index[0], df_preds.index[-1], freq = "30min")
        df_preds = pd.DataFrame(index = df_index).join(df_preds)

        # 48 steps per day
        horizons = np.tile(np.arange(48),int(len(df_preds.index)/48))
        df_preds['Horizon'] = horizons

        # We set up a folder for the given plant results
        path_out = f"reports/results/{plant}/deterministic/predictions"
        setup_folders(path_out)

        ### We save the predictions
        if args.tuned_run == True:
            filename_out = f'{path_out}/{plant}_{tech}_predictions_tuned.csv'
        else:
            filename_out = f'{path_out}/{plant}_{tech}_predictions.csv'
        df_preds.to_csv(filename_out, sep = ";")
        print("Predictions exported to:", filename_out)