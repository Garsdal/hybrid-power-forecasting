import pandas as pd
pd.options.plotting.backend = "plotly"
import numpy as np
import joblib
import os
import argparse

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
parser.add_argument('--plant', type = str, default = "HPP1")
parser.add_argument('--model', type = str, default = "LGB")
parser.add_argument('--n_sims', type = int, default = 200)
parser.add_argument('--max_uncertainty_wind', type = float, default = 0.1)
parser.add_argument('--max_uncertainty_solar', type = float, default = 0.1)
args = parser.parse_args()

def simulate_forecast_distribution(plant, model, max_uncertainty_wind, max_uncertainty_solar):
    if model == "LGB":
        features, targets = return_static_features(plant, "wind")
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        filename_wind = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features)}_{targets[0]}.sav"
        model_wind = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = True)
        model_wind.load_model(filename_wind)

        features, targets = return_static_features(plant, "solar")
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        filename_solar = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features)}_{targets[0]}.sav"
        model_solar = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = True)
        model_solar.load_model(filename_solar)

        features, targets = return_static_features(plant, "agr")
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
        filename_agr = f"{root_location}/{scratch_location}/models/LGB/{plant}/LGB_{plant}_static_numf{len(features)}_{targets[0]}.sav"
        model_agr = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = True)
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

        # We build non-sequential features to use Y_test later
        features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
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

        # We build non-sequential features to use Y_test later
        features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
        dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)

    # Simulation for loop #
    if (max_uncertainty_wind == 0) and (max_uncertainty_solar == 0):
        n = 1 #if we have no uncertainty we only run a single forecast for the distribution
    else:
        n = args.n_sims
    print("Total simulations for this run:", n)

    dict_wind = {}
    dict_solar = {}
    dict_agr = {}
    dict_agr_sum = {}
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

        # We pertubate the inputs  
        print("Adding noise...")  
        weights = create_weights(max_uncertainty_wind = max_uncertainty_wind, max_uncertainty_solar = max_uncertainty_solar)
        df_pertub = create_pertubed_inputs(df, uncertain_features, weights, seed = i+1)
        
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

        # Make predictions # 
        Y_pred_wind = model_wind.test(filename = '', set = "val")
        Y_pred_wind_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_wind, index = dt_val_padding)).values)
        Y_pred_wind_padded[Y_pred_wind_padded < 0] = 0  # remove negative predictions (only relevant for LSTM)

        Y_pred_solar = model_solar.test(filename = '', set = "val")
        Y_pred_solar_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_solar, index = dt_val_padding)).values)
        Y_pred_solar_padded[Y_pred_solar_padded < 0] = 0

        Y_pred_agr = model_agr.test(filename = '', set = "val")
        Y_pred_agr_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_agr, index = dt_val_padding)).values)
        Y_pred_agr_padded[Y_pred_agr_padded < 0] = 0

        Y_pred_agr_sum = (Y_pred_wind_padded*wind_cap + Y_pred_solar_padded*solar_cap)/agr_cap
        Y_pred_agr_sum[Y_pred_agr_sum < 0] = 0

        # We get the forecast errors
        Y_pred_agr_fc_error = Y_pred_agr_padded - Y_val
        Y_pred_agr_sum_fc_error = Y_pred_agr_sum - Y_val
        
        print("Saving simulations...")
        dict_wind[i] = Y_pred_wind_padded
        dict_solar[i] = Y_pred_solar_padded
        dict_agr[i] = Y_pred_agr_padded
        dict_agr_sum[i] = Y_pred_agr_sum
        dict_agr_fc_error[i] = Y_pred_agr_fc_error
        dict_agr_sum_fc_error[i] = Y_pred_agr_sum_fc_error

    # We create the dataframe with wind, solar, agr and agr_sum simulations | we insert a column with the true values
    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
    dt_train, dt_test, dt_val, X_train_wind, X_test_wind, X_val_wind, Y_train_wind, Y_test_wind, Y_val_wind, features_wind = build_features(df, features, targets)
    dict_wind['Y_true'] = Y_val_wind
    df_wind = pd.DataFrame(dict_wind, index = dt_val)

    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
    dt_train, dt_test, dt_val, X_train_solar, X_test_solar, X_val_solar, Y_train_solar, Y_test_solar, Y_val_solar, features_solar = build_features(df, features, targets)
    dict_solar['Y_true'] = Y_val_solar
    df_solar = pd.DataFrame(dict_solar, index = dt_val)

    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
    dt_train, dt_test, dt_val, X_train_agr, X_test_agr, X_val_agr, Y_train_agr, Y_test_agr, Y_val_agr, features_agr = build_features(df, features, targets)
    dict_agr['Y_true'] = Y_val_agr
    df_agr = pd.DataFrame(dict_agr, index = dt_val)
    dict_agr_sum['Y_true'] = Y_val_agr
    df_agr_sum = pd.DataFrame(dict_agr_sum, index = dt_val)

    # We pad all the simulations
    df_index = pd.date_range(df_agr.index[0], df_agr.index[-1], freq = "30min")
    df_agr = pd.DataFrame(index = df_index).join(df_agr)

    df_index = pd.date_range(df_agr_sum.index[0], df_agr_sum.index[-1], freq = "30min")
    df_agr_sum = pd.DataFrame(index = df_index).join(df_agr_sum)

    df_index = pd.date_range(df_wind.index[0], df_wind.index[-1], freq = "30min")
    df_wind = pd.DataFrame(index = df_index).join(df_wind)

    df_index = pd.date_range(df_solar.index[0], df_solar.index[-1], freq = "30min")
    df_solar = pd.DataFrame(index = df_index).join(df_solar)

    df_agr_fc_error = pd.DataFrame(dict_agr_fc_error, index = dt_val) # Forecast errors
    df_index = pd.date_range(df_agr_fc_error.index[0], df_agr_fc_error.index[-1], freq = "30min")
    df_agr_fc_error = pd.DataFrame(index = df_index).join(df_agr_fc_error)

    df_agr_sum_fc_error = pd.DataFrame(dict_agr_sum_fc_error, index = dt_val) # Forecast errors
    df_index = pd.date_range(df_agr_sum_fc_error.index[0], df_agr_sum_fc_error.index[-1], freq = "30min")
    df_agr_sum_fc_error = pd.DataFrame(index = df_index).join(df_agr_sum_fc_error)

    df_true_val_wind = pd.DataFrame(Y_val_wind, dt_val) # True values for validation set
    df_index = pd.date_range(df_true_val_wind.index[0], df_true_val_wind.index[-1], freq = "30min")
    df_true_val_wind = pd.DataFrame(index = df_index).join(df_true_val_wind)

    df_true_val_solar = pd.DataFrame(Y_val_solar, dt_val) # True values for validation set
    df_index = pd.date_range(df_true_val_solar.index[0], df_true_val_solar.index[-1], freq = "30min")
    df_true_val_solar = pd.DataFrame(index = df_index).join(df_true_val_solar)

    df_true_val_agr = pd.DataFrame(Y_val_agr, dt_val) # True values for validation set
    df_index = pd.date_range(df_true_val_agr.index[0], df_true_val_agr.index[-1], freq = "30min")
    df_true_val_agr = pd.DataFrame(index = df_index).join(df_true_val_agr)

    # We add horizons to all of our data to sort it by horizons
    df_wind['Horizon'] = np.tile(np.arange(48),int(len(df_wind.index)/48))
    df_solar['Horizon'] = np.tile(np.arange(48),int(len(df_solar.index)/48))
    df_agr['Horizon'] = np.tile(np.arange(48),int(len(df_agr.index)/48))
    df_agr_sum['Horizon'] = np.tile(np.arange(48),int(len(df_agr_sum.index)/48))

    df_true_val_wind['Horizon'] = np.tile(np.arange(48),int(len(df_true_val_wind)/48))
    df_true_val_solar['Horizon'] = np.tile(np.arange(48),int(len(df_true_val_solar)/48))
    df_true_val_agr['Horizon'] = np.tile(np.arange(48),int(len(df_true_val_agr)/48))

    df_agr_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_agr_fc_error)/48))
    df_agr_sum_fc_error['Horizon'] = np.tile(np.arange(48),int(len(df_agr_sum_fc_error)/48))

    # We concat all simulations and true values for each horizons
    dict_vals_wind = {}
    dict_vals_solar = {}
    dict_vals_agr = {}
    dict_vals_agr_sum = {}

    dict_vals_agr_fc_error = {}
    dict_vals_agr_sum_fc_error = {}

    # (this is for the validation set)
    dict_vals_true_wind = {}
    dict_vals_true_solar = {}
    dict_vals_true_agr = {}

    horizons = np.tile(np.arange(48),int(len(df_agr.index)/48))
    for horizon in np.unique(horizons):
        # We get the true values
        dict_vals_true_wind[horizon] = np.array(df_true_val_wind.loc[df_true_val_wind['Horizon'] == horizon])[:,-2]
        dict_vals_true_solar[horizon] = np.array(df_true_val_solar.loc[df_true_val_solar['Horizon'] == horizon])[:,-2]
        dict_vals_true_agr[horizon] = np.array(df_true_val_agr.loc[df_true_val_agr['Horizon'] == horizon])[:,-2]
        
        # We drop the last columns (Y_true, horizon) and keep all the values for the simulations
        dict_vals_wind[horizon] = np.concatenate(np.array(df_wind.loc[df_wind['Horizon'] == horizon])[:,:-2])
        dict_vals_solar[horizon] = np.concatenate(np.array(df_solar.loc[df_solar['Horizon'] == horizon])[:,:-2])
        dict_vals_agr[horizon] = np.concatenate(np.array(df_agr.loc[df_agr['Horizon'] == horizon])[:,:-2])
        dict_vals_agr_sum[horizon] = np.concatenate(np.array(df_agr_sum.loc[df_agr_sum['Horizon'] == horizon])[:,:-2])
        
        # We drop the last columns (Y_true, horizon) and keep all the values
        dict_vals_agr_fc_error[horizon] = np.concatenate(np.array(df_agr_fc_error.loc[df_agr_fc_error['Horizon'] == horizon])[:,:-2])
        dict_vals_agr_sum_fc_error[horizon] = np.concatenate(np.array(df_agr_sum_fc_error.loc[df_agr_sum_fc_error['Horizon'] == horizon])[:,:-2])
        
    df_vals_wind = pd.DataFrame(dict_vals_wind)
    df_vals_solar = pd.DataFrame(dict_vals_solar)
    df_vals_agr = pd.DataFrame(dict_vals_agr)
    df_vals_agr_sum = pd.DataFrame(dict_vals_agr_sum)

    df_vals_agr_fc_error = pd.DataFrame(dict_vals_agr_fc_error)
    df_vals_agr_sum_fc_error = pd.DataFrame(dict_vals_agr_sum_fc_error)

    df_vals_true_wind = pd.DataFrame(dict_vals_true_wind)
    df_vals_true_solar = pd.DataFrame(dict_vals_true_solar)
    df_vals_true_agr = pd.DataFrame(dict_vals_true_agr)

    # We get the standard deviations
    horizons = np.arange(48)
    std_wind_sims = pd.concat([df_vals_wind[horizons].std(), df_vals_true_wind[horizons].std()], axis = 1)
    std_wind_sims.columns = ['std_sim', 'std_true']

    std_solar_sims = pd.concat([df_vals_solar[horizons].std(), df_vals_true_solar[horizons].std()], axis = 1)
    std_solar_sims.columns = ['std_sim', 'std_true']

    # We get the means
    mean_wind_sims = pd.concat([df_vals_wind[horizons].mean(), df_vals_true_wind[horizons].mean()], axis = 1)
    mean_wind_sims.columns = ['mean_sim', 'mean_true']

    mean_solar_sims = pd.concat([df_vals_solar[horizons].mean(), df_vals_true_solar[horizons].mean()], axis = 1)
    mean_solar_sims.columns = ['mean_sim', 'mean_true']

    # The objective value is the distance from the true std for all horizons
    obj_val_wind = (np.square(std_wind_sims['std_true'] - std_wind_sims['std_sim'])).sum()
    obj_val_solar = (np.square(std_solar_sims['std_true'] - std_solar_sims['std_sim'])).sum()

    print("Plotting distributions...")
    #horizons = [0, 5, 11, 17, 23, 29, 35, 41, 47]
    horizons = [4, 12, 20, 28, 36, 42]
    fig_dist_wind = plot_simulated_violin_horizons(df_vals_true_wind, df_vals_wind, std_wind_sims, mean_wind_sims, max_uncertainty_wind, horizons)
    horizons = [16, 20, 24, 28, 32]
    fig_dist_solar = plot_simulated_violin_horizons(df_vals_true_solar, df_vals_solar, std_solar_sims, mean_solar_sims, max_uncertainty_solar, horizons, solar_plot = True)

    return obj_val_wind, obj_val_solar, fig_dist_wind, fig_dist_solar, mean_wind_sims, mean_solar_sims

# We set up folder structure for the given plant
plants = [args.plant]
models = [args.model]
for plant in plants:
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
        # We set up a folder for the given plant results
        path_out = f"reports/results/{plant}/probabilistic/distribution_simulation/optimization/{model}"
        setup_folders(path_out)

        root_location=os.path.abspath(os.sep)
        scratch_location='work3/s174440'

        # OPTIMIZATION
        #uncertainties = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] # HPP1/2/3 | LGB / LSTM
        uncertainties = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        objective_vals_wind = []
        objective_vals_solar = []
        for max_uncertainty in uncertainties:
            print("Simulating forecast distributions with max_uncertainty:", max_uncertainty)
            obj_val_wind, obj_val_solar, fig_dist_wind, fig_dist_solar, mean_wind_sims, mean_solar_sims = simulate_forecast_distribution(plant, model, max_uncertainty_wind = max_uncertainty, max_uncertainty_solar = max_uncertainty)

            # We save the means
            #mean_wind_sims.to_csv(f'{path_out}/mean_wind_sims_{max_uncertainty}.csv', sep = ";")
            #mean_solar_sims.to_csv(f'{path_out}/mean_solar_sims_{max_uncertainty}.csv', sep = ";")

            # We save the distribution
            fig_dist_wind.write_image(f"{path_out}/distribution_wind_{plant}_{model}_{max_uncertainty}_nsims_{args.n_sims}.png", width=1080, height=720) 
            fig_dist_solar.write_image(f"{path_out}/distribution_solar_{plant}_{model}_{max_uncertainty}_nsims_{args.n_sims}.png", width=1080, height=720) 

            # We save the objective values
            objective_vals_wind.append(obj_val_wind)
            objective_vals_solar.append(obj_val_solar)

        # PLOTTING
        df_obj_vals_wind = pd.DataFrame(objective_vals_wind, index = uncertainties)
        fig_obj_vals_wind = df_obj_vals_wind.plot()
        fig_obj_vals_wind.update_layout(template = 'ggplot2',
                        yaxis_title = 'MSE',
                        xaxis_title = 'Max Uncertainty')
        fig_obj_vals_wind.update_layout(title=go.layout.Title(text="Objective values (MSE) for error between std_true and std_sim for wind power across all horizons",xref="paper",x=0), showlegend=False)
        fig_obj_vals_wind.write_image(f"{path_out}/objective_values_wind_{plant}_{model}_nsims_{args.n_sims}.png", width=1080, height=720) 

        df_obj_vals_solar = pd.DataFrame(objective_vals_solar, index = uncertainties)
        fig_obj_vals_solar = df_obj_vals_solar.plot()
        fig_obj_vals_solar.update_layout(template = 'ggplot2',
                        yaxis_title = 'MSE',
                        xaxis_title = 'Max Uncertainty')
        fig_obj_vals_solar.update_layout(title=go.layout.Title(text="Objective values (MSE) for error between std_true and std_sim for solar power across all horizons",xref="paper",x=0), showlegend=False)
        fig_obj_vals_solar.write_image(f"{path_out}/objective_values_solar_{plant}_{model}_nsims_{args.n_sims}.png", width=1080, height=720)
