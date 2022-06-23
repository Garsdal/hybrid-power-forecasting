import pandas as pd
pd.options.plotting.backend = "plotly"
import numpy as np
import joblib
import os
import argparse
import json
import datetime

# Own imports
from src.utils import setup_folders, return_static_features, return_dynamic_features, create_pertubed_inputs, gen_gauss_copula, concat_day_simulations, create_quantiles, create_weights
from src.features.build_features import build_features, build_features_LSTM, build_features_seq
from src.models.predict_functions import recursive_predict_test_days, predict_full_sequence_days, recursive_predict_test_days_LSTM
from src.models.deterministic.models import RF, LR, LGB, Persistence, my_LSTM
from src.visualization.visualize import plot_quantile_bands, plot_simulated_violin_horizons, plot_single_day_simulations

# Plotting import
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors

# ARGPARSE | #Get plant + num simulations
parser = argparse.ArgumentParser()
parser.add_argument('--plant', type = str, default = "HPP2")
parser.add_argument('--model', type = str, default = "LGB")
parser.add_argument('--n_sims', type = int, default = 200)
parser.add_argument('--max_uncertainty_wind', type = float, default = 0.5)
parser.add_argument('--max_uncertainty_solar', type = float, default = 0.5)
parser.add_argument('--bias_corrected', type = bool, default = False)
args = parser.parse_args()

# We set up folder structure for the given plant
plants = [args.plant]
models = [args.model]
for plant in plants:
    root_location=os.path.abspath(os.sep)
    scratch_location='work3/s174440'

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

    # We include plant capacities (could be saved in a json that is not handed in)
    if plant == "Nazeerabad":
        wind_cap = 1
        solar_cap = 1
    elif plant == "HPP1":
        wind_cap = 1
        solar_cap = 1
    elif plant == "HPP2":
        wind_cap = 1
        solar_cap = 1
    elif plant == "HPP3":
        wind_cap = 1
        solar_cap = 1
    agr_cap = wind_cap + solar_cap

    # We load the models we want to use for predictions (we load them with placeholder X and Y values since we replace these when simulating)
    for model in models:
        print("Bias correction:", args.bias_corrected)
        # We set up a folder for the given plant results
        if args.bias_corrected:
            path_out = f"reports/results/{plant}/probabilistic/distribution_simulation/final/{model}/{args.max_uncertainty_wind}_{args.max_uncertainty_solar}/bias_corrected"
        else:
            path_out = f"reports/results/{plant}/probabilistic/distribution_simulation/final/{model}/{args.max_uncertainty_wind}_{args.max_uncertainty_solar}"
        setup_folders(path_out)

        if model == "LGB":
            features, targets = return_static_features(plant, "wind")
            dt_train, dt_test, dt_val, X_train_wind, X_test_wind, X_val_wind, Y_train_wind, Y_test_wind, Y_val_wind, features_wind = build_features(df, features, targets)
            filename_wind = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features_wind)}_{targets[0]}.sav"
            model_wind = LGB(dt_train, dt_test, dt_val, X_train_wind, X_test_wind, X_val_wind, Y_train_wind, Y_test_wind, Y_val_wind, features_wind, targets, static = True)
            model_wind.load_model(filename_wind)

            features, targets = return_static_features(plant, "solar")
            dt_train, dt_test, dt_val, X_train_solar, X_test_solar, X_val_solar, Y_train_solar, Y_test_solar, Y_val_solar, features_solar = build_features(df, features, targets)
            filename_solar = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features_solar)}_{targets[0]}.sav"
            model_solar = LGB(dt_train, dt_test, dt_val, X_train_solar, X_test_solar, X_val_solar, Y_train_solar, Y_test_solar, Y_val_solar, features_solar, targets, static = True)
            model_solar.load_model(filename_solar)

            features, targets = return_static_features(plant, "agr")
            dt_train, dt_test, dt_val, X_train_agr, X_test_agr, X_val_agr, Y_train_agr, Y_test_agr, Y_val_agr, features_agr = build_features(df, features, targets)
            filename_agr = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features_agr)}_{targets[0]}.sav"
            model_agr = LGB(dt_train, dt_test, dt_val, X_train_agr, X_test_agr, X_val_agr, Y_train_agr, Y_test_agr, Y_val_agr, features_agr, targets, static = True)
            model_agr.load_model(filename_agr)
        elif model == "RF":
            n_lag = 144; n_out = 48
            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_wind = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_dynamic_numf576_obs_power_wind(t).joblib"
            model_wind = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, seq = True)
            model_wind.load_model(filename_wind)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_solar = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_dynamic_numf576_obs_power_solar(t).joblib"
            model_solar = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, seq = True)
            model_solar.load_model(filename_solar)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_agr = f"{root_location}/{scratch_location}/models/RF/{plant}/RF_{plant}_dynamic_numf864_obs_power_agr(t).joblib"
            model_agr = RF(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = False, seq = True)
            model_agr.load_model(filename_agr)

            # We build non-sequential features to use Y_test and Y_val later
            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
            dt_train, dt_test, dt_val, X_train_wind, X_test_wind, X_val_wind, Y_train_wind, Y_test_wind, Y_val_wind, features_wind = build_features(df, features, targets)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
            dt_train, dt_test, dt_val, X_train_solar, X_test_solar, X_val_solar, Y_train_solar, Y_test_solar, Y_val_solar, features_solar = build_features(df, features, targets)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
            dt_train, dt_test, dt_val, X_train_agr, X_test_agr, X_val_agr, Y_train_agr, Y_test_agr, Y_val_agr, features_agr = build_features(df, features, targets)
        elif model == "LSTM":
            n_lag = 144; n_out = 48
            
            features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "wind")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_wind = f"{root_location}/{scratch_location}/models/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs20_{plant}_numf4_obs_power_wind.h5"
            model_wind = my_LSTM(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, recursive = False)
            model_wind.load_model(filename_wind)

            features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "solar")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_solar = f"{root_location}/{scratch_location}/models/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs20_{plant}_numf4_obs_power_solar.h5"
            model_solar = my_LSTM(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, recursive = False)
            model_solar.load_model(filename_solar) 
        
            features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "agr")
            dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
            filename_agr = f"{root_location}/{scratch_location}/models/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs20_{plant}_numf6_obs_power_agr.h5"
            model_agr = my_LSTM(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, recursive = False)
            model_agr.load_model(filename_agr)

            # We build non-sequential features to use Y_test and Y_val later
            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
            dt_train, dt_test, dt_val, X_train_wind, X_test_wind, X_val_wind, Y_train_wind, Y_test_wind, Y_val_wind, features_wind = build_features(df, features, targets)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
            dt_train, dt_test, dt_val, X_train_solar, X_test_solar, X_val_solar, Y_train_solar, Y_test_solar, Y_val_solar, features_solar = build_features(df, features, targets)

            features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
            dt_train, dt_test, dt_val, X_train_agr, X_test_agr, X_val_agr, Y_train_agr, Y_test_agr, Y_val_agr, features_agr = build_features(df, features, targets)

        # Simulation for loop #
        n = args.n_sims
        dict_wind = {}
        dict_solar = {}
        dict_agr = {}
        dict_agr_sum = {}

        dict_wind_fc_error = {}
        dict_solar_fc_error = {}
        dict_agr_fc_error = {}
        dict_agr_sum_fc_error = {}
        for i in range(n):
            if i % 50 == 0:
                print("run:", i, "plant:", plant, "model:", model)

            # We have different uncertain columns
            if plant in ["HPP1", "HPP2", "HPP3"]:
                uncertain_features = ['WINDSPEED_100m', 'DSWRF']
            else:
                uncertain_features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_ghi_4km_NIWE']

            # The very first simulation is deterministic
            if i == 0:
                print("Creating deterministic forecast with no noise...")
                df_pertub = df
            else:
                # We pertubate the inputs  
                print("Adding noise...")  
                weights = create_weights(max_uncertainty_wind = args.max_uncertainty_wind, max_uncertainty_solar = args.max_uncertainty_solar)
                df_pertub = create_pertubed_inputs(df, uncertain_features, weights, seed = i+1)

            # We save .describe of the the data before and after adding noise
            df[uncertain_features].describe().to_csv(f'{path_out}/df_describe.csv', sep = ";")
            df_pertub[uncertain_features].describe().to_csv(f'{path_out}/df_pertub_describe.csv', sep = ";")

            # We create the input features from the pertubated dataframe
            if model == "LGB":
                features, targets = return_static_features(plant, "wind")
                dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                model_wind.X_test = X_test
                model_wind.X_val = X_val

                features, targets = return_static_features(plant, "solar")
                dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                model_solar.X_test = X_test
                model_solar.X_val = X_val

                features, targets = return_static_features(plant, "agr")
                dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                model_agr.X_test = X_test
                model_agr.X_val = X_val

                dt_test_padding = dt_test # we save the test datetimes for sequences for padding
                dt_val_padding = dt_val

            elif model == "RF":
                n_lag = 144; n_out = 48

                features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
                dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_wind.X_test = X_test_seq
                model_wind.X_val = X_val_seq

                features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
                dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_solar.X_test = X_test_seq
                model_solar.X_val = X_val_seq

                features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
                dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_agr.X_test = X_test_seq
                model_agr.X_val = X_val_seq

                dt_test_padding = dt_test_seq # we save the test datetimes for sequences for padding
                dt_val_padding = dt_val_seq

            elif model == "LSTM":
                n_lag = 144; n_out = 48

                features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "wind")
                dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_wind.X_test = X_test_LSTM
                model_wind.X_val = X_val_LSTM

                features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "solar")
                dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_solar.X_test = X_test_LSTM
                model_solar.X_val = X_val_LSTM

                features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "agr")
                dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                model_agr.X_test = X_test_LSTM
                model_agr.X_val = X_val_LSTM

                dt_test_padding = dt_test_LSTM # we save the test datetimes for sequences for padding
                dt_val_padding = dt_val_LSTM

            # Make predictions validation set # 
            print("Predictions for val set...") 
            Y_pred_wind_val = model_wind.test(filename = '', set = "val")
            Y_pred_wind_val_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_wind_val, index = dt_val_padding)).values)
            
            Y_pred_solar_val = model_solar.test(filename = '', set = "val")
            Y_pred_solar_val_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_solar_val, index = dt_val_padding)).values)
            
            Y_pred_agr_val = model_agr.test(filename = '', set = "val")
            Y_pred_agr_val_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_agr_val, index = dt_val_padding)).values)
            
            Y_pred_agr_sum_val = (Y_pred_wind_val_padded*wind_cap + Y_pred_solar_val_padded*solar_cap)/agr_cap

            # We clip predictions at 0
            Y_pred_wind_val_padded[Y_pred_wind_val_padded < 0] = 0
            Y_pred_solar_val_padded[Y_pred_solar_val_padded < 0] = 0
            Y_pred_agr_val_padded[Y_pred_agr_val_padded < 0] = 0
            Y_pred_agr_sum_val[Y_pred_agr_sum_val < 0] = 0

            # We get the forecast errors from the validation set #
            Y_pred_wind_fc_error = Y_val_wind - Y_pred_wind_val_padded
            Y_pred_solar_fc_error = Y_val_solar - Y_pred_solar_val_padded
            Y_pred_agr_fc_error = Y_val_agr - Y_pred_agr_val_padded
            Y_pred_agr_sum_fc_error = Y_val_agr - Y_pred_agr_sum_val
            
            # Make predictions test set #
            print("Predictions for test set...") 
            Y_pred_wind_test = model_wind.test(filename = '', set = "test")
            Y_pred_wind_test_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_wind_test, index = dt_test_padding)).values)
            
            Y_pred_solar_test = model_solar.test(filename = '', set = "test")
            Y_pred_solar_test_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_solar_test, index = dt_test_padding)).values)
            
            Y_pred_agr_test = model_agr.test(filename = '', set = "test")
            Y_pred_agr_test_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_agr_test, index = dt_test_padding)).values)
            
            Y_pred_agr_sum_test = (Y_pred_wind_test_padded*wind_cap + Y_pred_solar_test_padded*solar_cap)/agr_cap

            # We clip predictions at 0
            Y_pred_wind_test_padded[Y_pred_wind_test_padded < 0] = 0
            Y_pred_solar_test_padded[Y_pred_solar_test_padded < 0] = 0
            Y_pred_agr_test_padded[Y_pred_agr_test_padded < 0] = 0
            Y_pred_agr_sum_test[Y_pred_agr_sum_test < 0] = 0

            print("Saving simulations...")
            dict_wind[i] = Y_pred_wind_test_padded
            dict_solar[i] = Y_pred_solar_test_padded
            dict_agr[i] = Y_pred_agr_test_padded
            dict_agr_sum[i] = Y_pred_agr_sum_test

            dict_wind_fc_error[i] = Y_pred_wind_fc_error
            dict_solar_fc_error[i] = Y_pred_solar_fc_error
            dict_agr_fc_error[i] = Y_pred_agr_fc_error
            dict_agr_sum_fc_error[i] = Y_pred_agr_sum_fc_error

        # We create the dataframe with wind, solar, agr and agr_sum simulations
        dict_wind['Y_true'] = Y_test_wind
        df_wind = pd.DataFrame(dict_wind, index = dt_test)

        dict_solar['Y_true'] = Y_test_solar
        df_solar = pd.DataFrame(dict_solar, index = dt_test)

        dict_agr['Y_true'] = Y_test_agr
        df_agr = pd.DataFrame(dict_agr, index = dt_test)

        dict_agr_sum['Y_true'] = Y_test_agr
        df_agr_sum = pd.DataFrame(dict_agr_sum, index = dt_test)

        df_wind_fc_error = pd.DataFrame(dict_wind_fc_error, index = dt_val)
        df_solar_fc_error = pd.DataFrame(dict_solar_fc_error, index = dt_val)
        df_agr_fc_error = pd.DataFrame(dict_agr_fc_error, index = dt_val)
        df_agr_sum_fc_error = pd.DataFrame(dict_agr_sum_fc_error, index = dt_val)

        # We pad all the simulations
        df_index = pd.date_range(df_agr.index[0], df_agr.index[-1], freq = "30min")
        df_agr = pd.DataFrame(index = df_index).join(df_agr)

        df_index = pd.date_range(df_agr_sum.index[0], df_agr_sum.index[-1], freq = "30min")
        df_agr_sum = pd.DataFrame(index = df_index).join(df_agr_sum)

        df_index = pd.date_range(df_wind.index[0], df_wind.index[-1], freq = "30min")
        df_wind = pd.DataFrame(index = df_index).join(df_wind)

        df_index = pd.date_range(df_solar.index[0], df_solar.index[-1], freq = "30min")
        df_solar = pd.DataFrame(index = df_index).join(df_solar)

        df_index = pd.date_range(df_wind_fc_error.index[0], df_wind_fc_error.index[-1], freq = "30min")
        df_wind_fc_error = pd.DataFrame(index = df_index).join(df_wind_fc_error)

        df_index = pd.date_range(df_solar_fc_error.index[0], df_solar_fc_error.index[-1], freq = "30min")
        df_solar_fc_error = pd.DataFrame(index = df_index).join(df_solar_fc_error)

        df_index = pd.date_range(df_agr_fc_error.index[0], df_agr_fc_error.index[-1], freq = "30min")
        df_agr_fc_error = pd.DataFrame(index = df_index).join(df_agr_fc_error)

        df_index = pd.date_range(df_agr_sum_fc_error.index[0], df_agr_sum_fc_error.index[-1], freq = "30min")
        df_agr_sum_fc_error = pd.DataFrame(index = df_index).join(df_agr_sum_fc_error)

        df_true_test_wind = pd.DataFrame(Y_test_wind, dt_test) # True values for validation set
        df_index = pd.date_range(df_true_test_wind.index[0], df_true_test_wind.index[-1], freq = "30min")
        df_true_test_wind = pd.DataFrame(index = df_index).join(df_true_test_wind)

        df_true_test_solar = pd.DataFrame(Y_test_solar, dt_test) # True values for validation set
        df_index = pd.date_range(df_true_test_solar.index[0], df_true_test_solar.index[-1], freq = "30min")
        df_true_test_solar = pd.DataFrame(index = df_index).join(df_true_test_solar)

        df_true_test_agr = pd.DataFrame(Y_test_agr, dt_test) # True values for validation set
        df_index = pd.date_range(df_true_test_agr.index[0], df_true_test_agr.index[-1], freq = "30min")
        df_true_test_agr = pd.DataFrame(index = df_index).join(df_true_test_agr)

        # We add horizons to all of our data to sort it by horizons
        df_wind['Horizon'] = np.tile(np.arange(48),int(len(df_wind.index)/48))
        df_solar['Horizon'] = np.tile(np.arange(48),int(len(df_solar.index)/48))
        df_agr['Horizon'] = np.tile(np.arange(48),int(len(df_agr.index)/48))
        df_agr_sum['Horizon'] = np.tile(np.arange(48),int(len(df_agr_sum.index)/48))

        df_true_test_wind['Horizon'] = np.tile(np.arange(48),int(len(df_true_test_wind)/48))
        df_true_test_solar['Horizon'] = np.tile(np.arange(48),int(len(df_true_test_solar)/48))
        df_true_test_agr['Horizon'] = np.tile(np.arange(48),int(len(df_true_test_agr)/48))

        df_wind_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_wind_fc_error)/48))
        df_solar_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_solar_fc_error)/48))
        df_agr_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_agr_fc_error)/48))
        df_agr_sum_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_agr_sum_fc_error)/48))

        # We concat all simulations and true values for each horizons
        dict_vals_wind = {}
        dict_vals_solar = {}
        dict_vals_agr = {}
        dict_vals_agr_sum = {}

        dict_vals_wind_fc_error = {}
        dict_vals_solar_fc_error = {}
        dict_vals_agr_fc_error = {}
        dict_vals_agr_sum_fc_error = {}

        # (this is for the test set)
        dict_vals_true_wind = {}
        dict_vals_true_solar = {}
        dict_vals_true_agr = {}

        horizons = np.tile(np.arange(48),int(len(df_agr.index)/48))
        for horizon in np.unique(horizons):
            # We get the true values
            dict_vals_true_wind[horizon] = np.array(df_true_test_wind.loc[df_true_test_wind['Horizon'] == horizon])[:,-2]
            dict_vals_true_solar[horizon] = np.array(df_true_test_solar.loc[df_true_test_solar['Horizon'] == horizon])[:,-2]
            dict_vals_true_agr[horizon] = np.array(df_true_test_agr.loc[df_true_test_agr['Horizon'] == horizon])[:,-2]
            
            # We drop the last columns (Y_true, horizon) and keep all the values for the simulations
            dict_vals_wind[horizon] = np.concatenate(np.array(df_wind.loc[df_wind['Horizon'] == horizon])[:,:-2])
            dict_vals_solar[horizon] = np.concatenate(np.array(df_solar.loc[df_solar['Horizon'] == horizon])[:,:-2])
            dict_vals_agr[horizon] = np.concatenate(np.array(df_agr.loc[df_agr['Horizon'] == horizon])[:,:-2])
            dict_vals_agr_sum[horizon] = np.concatenate(np.array(df_agr_sum.loc[df_agr_sum['Horizon'] == horizon])[:,:-2])
            
            # We drop the last columns (Y_true, horizon) and keep all the values
            dict_vals_wind_fc_error[horizon] = np.concatenate(np.array(df_wind_fc_error.loc[df_wind_fc_error['Horizon'] == horizon])[:,:-2])
            dict_vals_solar_fc_error[horizon] = np.concatenate(np.array(df_solar_fc_error.loc[df_solar_fc_error['Horizon'] == horizon])[:,:-2])
            dict_vals_agr_fc_error[horizon] = np.concatenate(np.array(df_agr_fc_error.loc[df_agr_fc_error['Horizon'] == horizon])[:,:-2])
            dict_vals_agr_sum_fc_error[horizon] = np.concatenate(np.array(df_agr_sum_fc_error.loc[df_agr_sum_fc_error['Horizon'] == horizon])[:,:-2])
            
        # Convert simulations to dataframe
        df_vals_wind = pd.DataFrame(dict_vals_wind)
        df_vals_solar = pd.DataFrame(dict_vals_solar)
        df_vals_agr = pd.DataFrame(dict_vals_agr)
        df_vals_agr_sum = pd.DataFrame(dict_vals_agr_sum)

        df_vals_wind_fc_error = pd.DataFrame(dict_vals_wind_fc_error)
        df_vals_solar_fc_error = pd.DataFrame(dict_vals_solar_fc_error)
        df_vals_agr_fc_error = pd.DataFrame(dict_vals_agr_fc_error)
        df_vals_agr_sum_fc_error = pd.DataFrame(dict_vals_agr_sum_fc_error)

        df_vals_true_wind = pd.DataFrame(dict_vals_true_wind)
        df_vals_true_solar = pd.DataFrame(dict_vals_true_solar)
        df_vals_true_agr = pd.DataFrame(dict_vals_true_agr)

        # Bias correction for the forecast error distributions
        if args.bias_corrected:
            df_vals_wind_fc_error = df_vals_wind_fc_error - df_vals_wind_fc_error.mean()
            df_vals_solar_fc_error = df_vals_solar_fc_error - df_vals_solar_fc_error.mean()
            df_vals_agr_fc_error = df_vals_agr_fc_error - df_vals_agr_fc_error.mean()
            df_vals_agr_sum_fc_error = df_vals_agr_sum_fc_error - df_vals_agr_sum_fc_error.mean()

        # We get the standard deviations for wind/solar/agr/agr_sum simulations
        horizons = np.arange(48)
        std_wind_sims = pd.concat([df_vals_wind[horizons].std(), df_vals_true_wind[horizons].std()], axis = 1)
        std_wind_sims.columns = ['std_sim', 'std_true']

        std_solar_sims = pd.concat([df_vals_solar[horizons].std(), df_vals_true_solar[horizons].std()], axis = 1)
        std_solar_sims.columns = ['std_sim', 'std_true']

        std_agr_sims = pd.concat([df_vals_agr[horizons].std(), df_vals_true_agr[horizons].std()], axis = 1)
        std_agr_sims.columns = ['std_sim', 'std_true']

        std_agr_sum_sims = pd.concat([df_vals_agr_sum[horizons].std(), df_vals_true_agr[horizons].std()], axis = 1)
        std_agr_sum_sims.columns = ['std_sim', 'std_true']

        std_hybrid_sims = pd.concat([df_vals_agr_sum[horizons].std(), df_vals_agr[horizons].std()], axis = 1)
        std_hybrid_sims.columns = ['std_agr', 'std_agr_sum']

        std_fc_error_sims = pd.concat([df_vals_agr_fc_error[horizons].std(), df_vals_agr_sum_fc_error[horizons].std()], axis = 1)
        std_fc_error_sims.columns = ['std_agr', 'std_agr_sum']

        # We get the means for wind/solar/agr/agr_sum simulations
        mean_wind_sims = pd.concat([df_vals_wind[horizons].mean(), df_vals_true_wind[horizons].mean()], axis = 1)
        mean_wind_sims.columns = ['mean_sim', 'mean_true']

        mean_solar_sims = pd.concat([df_vals_solar[horizons].mean(), df_vals_true_solar[horizons].mean()], axis = 1)
        mean_solar_sims.columns = ['mean_sim', 'mean_true']

        mean_agr_sims = pd.concat([df_vals_agr[horizons].mean(), df_vals_true_agr[horizons].mean()], axis = 1)
        mean_agr_sims.columns = ['mean_sim', 'mean_true']

        mean_agr_sum_sims = pd.concat([df_vals_agr_sum[horizons].mean(), df_vals_true_agr[horizons].mean()], axis = 1)
        mean_agr_sum_sims.columns = ['mean_sim', 'mean_true']

        mean_hybrid_sims = pd.concat([df_vals_agr_sum[horizons].mean(), df_vals_agr[horizons].mean()], axis = 1)
        mean_hybrid_sims.columns = ['mean_agr', 'mean_agr_sum']

        mean_fc_error_sims = pd.concat([df_vals_agr_fc_error[horizons].mean(), df_vals_agr_sum_fc_error[horizons].mean()], axis = 1)
        mean_fc_error_sims.columns = ['mean_agr', 'mean_agr_sum']

        # We plot the distribution of simulated forecasts for the given uncertainty level
        print("Plotting distributions...")
        horizons = [0, 5, 11, 17, 23, 29, 35, 41, 47]
        fig_dist_wind = plot_simulated_violin_horizons(['Y_wind_true', 'Y_wind'], df_vals_true_wind, df_vals_wind, std_wind_sims, mean_wind_sims, horizons)
        fig_dist_wind.write_image(f"{path_out}/distribution_wind_{plant}_{model}.pdf", width=1080, height=450) 

        horizons = [17, 19, 21, 23, 25, 27, 29, 31, 33]
        fig_dist_solar = plot_simulated_violin_horizons(['Y_solar_true', 'Y_solar'], df_vals_true_solar, df_vals_solar, std_solar_sims, mean_solar_sims, horizons, solar_plot = True)
        fig_dist_solar.write_image(f"{path_out}/distribution_solar_{plant}_{model}.pdf", width=1080, height=450) 

        horizons = [0, 5, 11, 17, 23, 29, 35, 41, 47]
        fig_dist_agr = plot_simulated_violin_horizons(['Y_agr_true', 'Y_agr'], df_vals_true_agr, df_vals_agr, std_agr_sims, mean_agr_sims, horizons)
        fig_dist_agr.write_image(f"{path_out}/distribution_agr_{plant}_{model}.pdf", width=1080, height=450) 

        fig_dist_agr_sum = plot_simulated_violin_horizons(['Y_agr_true', 'Y_agr_sum'], df_vals_true_agr, df_vals_agr_sum, std_agr_sum_sims, mean_agr_sum_sims, horizons)
        fig_dist_agr_sum.write_image(f"{path_out}/distribution_agr_sum_{plant}_{model}.pdf", width=1080, height=450) 

        # We plot the distributions of agr and agr_sum residuals across horizons
        fig_dist_fc_error = plot_simulated_violin_horizons(['Y_agr', 'Y_agr_sum'], df_vals_agr_fc_error, df_vals_agr_sum_fc_error, std_fc_error_sims, mean_fc_error_sims, horizons, yaxis_title = "Forecast error")
        fig_dist_fc_error.write_image(f"{path_out}/distribution_fc_error_agr_comparison_{plant}_{model}.pdf", width=1080, height=450)

        # We plot the distribution of agr and agr_sum forecast values against each other
        fig_dist_agr_comparison = plot_simulated_violin_horizons(['Y_agr', 'Y_agr_sum'], df_vals_agr, df_vals_agr_sum, std_hybrid_sims, mean_hybrid_sims, horizons)
        fig_dist_agr_comparison.write_image(f"{path_out}/distribution_agr_comparison_{plant}_{model}.pdf", width=1080, height=450)

        if plant in ["HPP1"]:
            #day = '2019-08-18' #val
            day = '2019-08-16'
        elif plant == "HPP2":
            #day = '2021-02-21' #val
            day = '2021-02-19'
        elif plant == "HPP3":
            #day = '2018-12-30' #val
            day = '2018-12-28'
        elif plant == "Nazeerabad":
            #day = '2018-11-25' #val
            day = '2018-11-23'

        # We plot single days of all simulations to compare
        fig_day_wind = plot_single_day_simulations(df_wind, day, "Simulated wind forecasts for a single day")
        fig_day_wind.write_image(f"{path_out}/day_simulations_wind_{plant}_{model}.pdf", width=720, height=720) 

        fig_day_solar = plot_single_day_simulations(df_solar, day, "Simulated solar forecasts for a single day")
        fig_day_solar.write_image(f"{path_out}/day_simulations_solar_{plant}_{model}.pdf", width=720, height=720) 

        fig_day_agr = plot_single_day_simulations(df_agr, day, "Simulated solar forecasts for a single day")
        fig_day_agr.write_image(f"{path_out}/day_simulations_agr_{plant}_{model}.pdf", width=720, height=720) 

        fig_day_agr_sum = plot_single_day_simulations(df_agr_sum, day, "Simulated solar forecasts for a single day")
        fig_day_agr_sum.write_image(f"{path_out}/day_simulations_agr_sum_{plant}_{model}.pdf", width=720, height=720) 

        # We only continue with the quantile results if we have more than 1 simulation e.g. columns > 3 (including horizon/y_true)
        if len(df_wind.columns) > 3:
            # We get quantiles from the forecast errors
            df_wind_fc_error_quantiles = create_quantiles(df_wind, df_vals_wind_fc_error)
            df_solar_fc_error_quantiles = create_quantiles(df_solar, df_vals_solar_fc_error)
            df_agr_fc_error_quantiles = create_quantiles(df_agr, df_vals_agr_fc_error)
            df_agr_sum_fc_error_quantiles = create_quantiles(df_agr_sum, df_vals_agr_sum_fc_error)

            # fc error bands plot
            fig_wind_bands_fc_error = plot_quantile_bands(df_wind, df_wind_fc_error_quantiles, day, residual_quantile = True)
            fig_wind_bands_fc_error.write_image(f"{path_out}/bands_wind_fc_error_{plant}_{model}.pdf", width=720, height=720)

            fig_solar_bands_fc_error = plot_quantile_bands(df_solar, df_solar_fc_error_quantiles, day, residual_quantile = True)
            fig_solar_bands_fc_error.write_image(f"{path_out}/bands_solar_fc_error_{plant}_{model}.pdf", width=720, height=720)
            
            fig_agr_bands_fc_error = plot_quantile_bands(df_agr, df_agr_fc_error_quantiles, day, residual_quantile = True)
            fig_agr_bands_fc_error.write_image(f"{path_out}/bands_agr_fc_error_{plant}_{model}.pdf", width=720, height=720)

            fig_agr_sum_bands_fc_error = plot_quantile_bands(df_agr_sum, df_agr_sum_fc_error_quantiles, day, residual_quantile = True)
            fig_agr_sum_bands_fc_error.write_image(f"{path_out}/bands_agr_sum_fc_error_{plant}_{model}.pdf", width=720, height=720)
                
            # we want to get the quantiles and compare the number of points from
            def check_bands(df, df_quantiles):
                keys = np.array(df_quantiles.columns)
                n_bands = int(len(df_quantiles.columns)/2)

                Y_pred = df[0]
                Y_true = df[['Y_true', 'Horizon']]

                dict_check = {}
                dict_check_horizon = {}
                dict_PINAW = {}
                for cnt in range(0, n_bands):
                    lower_quant = df_quantiles[keys[cnt]].name
                    upper_quant = df_quantiles[keys[-(cnt+1)]].name

                    print("Lower quant:", lower_quant, "Upper quant:", upper_quant)
                    print("Checking values...")

                    # We have to check how many values are within the quantiles for each horizon
                    cnt_inside = 0
                    cnt_total = 0
                    #dict_check_horizon[upper_quant] = {}
                    dict_check_horizon[upper_quant] = []
                    dict_PINAW[upper_quant] = []
                    for horizon in np.unique(Y_true['Horizon']):
                        Y_pred_horizon = df.loc[df['Horizon'] == horizon][0]
                        Y_true_horizon =df.loc[df['Horizon'] == horizon]['Y_true']

                        idx = np.array(Y_true_horizon.isna())
                        Y_pred_horizon = np.array(Y_pred_horizon[~idx].values)
                        Y_true_horizon = np.array(Y_true_horizon[~idx].values)

                        lower_quant_vals = Y_pred_horizon + df_quantiles.loc[horizon, lower_quant]
                        upper_quant_vals = Y_pred_horizon + df_quantiles.loc[horizon, upper_quant]

                        # We check how many values are within the band for each horizon
                        # Referred to as the prediction interval coverage probability score (PICP)
                        values_within_band = np.sum((lower_quant_vals <= Y_true_horizon) & (Y_true_horizon <= upper_quant_vals))
                        values_total_band = Y_true_horizon.shape[0]

                        cnt_total = cnt_total + values_total_band
                        cnt_inside = cnt_inside + values_within_band

                        print("Values inside:", cnt_inside, "out of total:", cnt_total)
                        #dict_check_horizon[upper_quant][str(horizon)] = values_within_band/values_total_band 
                        dict_check_horizon[upper_quant].append(values_within_band/values_total_band)

                        # We calculate the prediction interval normalized average width (PINAW)
                        H = Y_pred_horizon.shape[0]
                        Y_max = np.nanmax(Y_pred_horizon)
                        Y_min = np.nanmin(Y_pred_horizon)
                        U = upper_quant_vals
                        L = lower_quant_vals
                        PINAW = 1/(H*(Y_max - Y_min)) * np.nansum(U-L)
                        dict_PINAW[upper_quant].append(PINAW)

                    # We save the ratio for the given quantile
                    dict_check[upper_quant] = cnt_inside/cnt_total

                return dict_check, dict_check_horizon, dict_PINAW

            dict_PICP_wind, dict_PICP_wind_horizons, dict_PINAW_wind = check_bands(df_wind, df_wind_fc_error_quantiles)
            with open(f'{path_out}/quantile_PICP_wind.json', 'w') as outfile:
                json.dump(dict_PICP_wind, outfile, indent = 2)
            with open(f'{path_out}/quantile_PICP_wind_horizons.json', 'w') as outfile:
                json.dump(dict_PICP_wind_horizons, outfile, indent = 3)
            with open(f'{path_out}/quantile_PINAW_wind_horizons.json', 'w') as outfile:
                json.dump(dict_PINAW_wind, outfile, indent = 3)

            dict_PICP_solar, dict_PICP_solar_horizons, dict_PINAW_solar = check_bands(df_solar, df_solar_fc_error_quantiles)
            with open(f'{path_out}/quantile_PICP_solar.json', 'w') as outfile:
                json.dump(dict_PICP_solar, outfile, indent = 2)
            with open(f'{path_out}/quantile_PICP_solar_horizons.json', 'w') as outfile:
                json.dump(dict_PICP_solar_horizons, outfile, indent = 3)
            with open(f'{path_out}/quantile_PINAW_solar_horizons.json', 'w') as outfile:
                json.dump(dict_PINAW_solar, outfile, indent = 3)
            
            dict_PICP_agr, dict_PICP_agr_horizons, dict_PINAW_agr = check_bands(df_agr, df_agr_fc_error_quantiles)
            with open(f'{path_out}/quantile_PICP_agr.json', 'w') as outfile:
                json.dump(dict_PICP_agr, outfile, indent = 2)
            with open(f'{path_out}/quantile_PICP_agr_horizons.json', 'w') as outfile:
                json.dump(dict_PICP_agr_horizons, outfile, indent = 3)
            with open(f'{path_out}/quantile_PINAW_agr_horizons.json', 'w') as outfile:
                json.dump(dict_PINAW_agr, outfile, indent = 3)
            
            dict_PICP_agr_sum, dict_PICP_agr_sum_horizons, dict_PINAW_agr_sum = check_bands(df_agr_sum, df_agr_sum_fc_error_quantiles)
            with open(f'{path_out}/quantile_PICP_agr_sum.json', 'w') as outfile:
                json.dump(dict_PICP_agr_sum, outfile, indent = 2)
            with open(f'{path_out}/quantile_PICP_agr_sum_horizons.json', 'w') as outfile:
                json.dump(dict_PICP_agr_sum_horizons, outfile, indent = 3)
            with open(f'{path_out}/quantile_PINAW_agr_sum_horizons.json', 'w') as outfile:
                json.dump(dict_PINAW_agr_sum, outfile, indent = 3)

            ### Residual distribution plot ##
            # We create a plot with residual distributions for all validation days for a given horizon = 0 #
            horizon_select = 24

            # We get all forecast days 
            forecast_days = dt_val[dt_val.time == datetime.time(0, 0)]

            dict_residuals_val_days = {}
            # For loop over fc days
            for day in forecast_days:
                day_format = day.strftime('%Y-%m-%d')

                # First we subset for a given forecast day
                day_residuals = df_agr_fc_error.loc[day_format]

                # We then grab values for a given horizons
                idx = day_residuals['Horizon'] == horizon_select
                day_residuals_vals = day_residuals[idx].drop(['Horizon'], axis = 1)

                # We then grab all residuals for horizon = 0 (n = n_sims) and save them in a dict, the key is the fc day
                dict_residuals_val_days[day_format] = np.concatenate(day_residuals_vals.values)

            df_residuals_days = pd.DataFrame(dict_residuals_val_days)

            # We create a ridgeplot to see if the distributions of residuals change with validation days
            show_legend = [True] + list(np.repeat(False, 250, axis=0))
            forecast_days = df_residuals_days.columns[0:10]
            colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(forecast_days), colortype='rgb')
            
            fig = go.Figure()
            for day, color in zip(forecast_days, colors):
                data_line = df_residuals_days[day].values
                fig.add_trace(go.Violin(x=data_line, line_color=color, showlegend = False))
                
            fig.update_traces(orientation='h', side='positive', width=3, points=False)
            fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
            fig.update_layout(#template = 'ggplot2',
                                yaxis_title = f'Forecast days (horizon = {horizon_select})')
            fig.update_layout(legend_title="", font=dict(family="Arial",size=18))
            
            fig.write_image(f"{path_out}/ridgeplot_residual_distributions_new_{plant}_{model}.pdf", width=1080, height=720)

            ### Final PINAW width plot ###
            quantile = 0.95
            df_PINAW = pd.DataFrame({'PINAW_agr': dict_PINAW_agr[quantile], 'PINAW_agr_sum': dict_PINAW_agr_sum[quantile]}, index = np.arange(48))
            fig_PINAW = df_PINAW.plot.bar(barmode = 'group')
            fig_PINAW.update_layout(#template = 'ggplot2',
                            yaxis_title = 'PINAW',
                            xaxis_title = 'Horizons (30min resolution)')
            fig_PINAW.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.01))
            fig_PINAW.update_layout(legend_title="", font=dict(family="Arial",size=18))
            fig_PINAW.write_image(f"{path_out}/barplot_PINAW_{plant}_{model}.pdf", width=1080, height=450)

            quantile = 0.95
            df_PICP = pd.DataFrame({'PICP_agr': dict_PICP_agr_horizons[quantile], 'PICP_agr_sum': dict_PICP_agr_sum_horizons[quantile]}, index = np.arange(48))
            fig_PICP = df_PICP.plot.bar(barmode = 'group')
            fig_PICP.update_layout(#template = 'ggplot2',
                    yaxis_title = 'PICP',
                    xaxis_title = 'Horizons (30min resolution)')
            fig_PICP.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.01))
            fig_PICP.update_layout(legend_title="", font=dict(family="Arial",size=18))
            fig_PICP.add_hline(y=quantile, line_width=1, line_dash="dash", line_color="black")
            fig_PICP.write_image(f"{path_out}/barplot_PICP_{plant}_{model}.pdf", width=1080, height=450)