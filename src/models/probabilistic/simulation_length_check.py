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
from src.visualization.visualize import plot_quantile_bands

# Plotting import
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors

# ARGPARSE | #Get plant + num simulations
parser = argparse.ArgumentParser()
parser.add_argument('--plant', type = str, default = "HPP3")
parser.add_argument('--model', type = str, default = "RF")
args = parser.parse_args()

# We simulate different number of forecasts and save the maximum standard deviation
max_std_list_wind, mean_std_list_wind = [], []
max_std_list_solar, mean_std_list_solar = [], []
max_std_list_agr, mean_std_list_agr = [], []
max_std_list_agr_sum, mean_std_list_agr_sum = [], []

#n_sims = np.arange(1, 75)
#n_sims = np.arange(10, 210, 10)
#n_sims = [300, 400, 500]
n_sims = [1000]
for n_sim in n_sims:
    print("Simulations in this run:", n_sim)
    # We set up folder structure for the given plant
    plants = [args.plant]
    models = [args.model]
    for plant in plants:
        # We set up a folder for the given plant results
        path_out = f"reports/results/{plant}/probabilistic"
        setup_folders(path_out)

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
            n = n_sim
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
                weights = create_weights(max_uncertainty = 0.1)
                df_pertub = create_pertubed_inputs(df, uncertain_features, weights, seed = i+1)
                
                # We create the input features from the pertubated dataframe
                if model == "LGB":
                    features, targets = return_static_features(plant, "wind")
                    dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                    model_wind.X_test = X_test # we update the values in the model object

                    features, targets = return_static_features(plant, "solar")
                    dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                    model_solar.X_test = X_test

                    features, targets = return_static_features(plant, "agr")
                    dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df_pertub, features, targets)
                    model_agr.X_test = X_test

                    dt_test_padding = dt_test # we save the test datetimes for sequences for padding

                elif model == "RF":
                    n_lag = 144; n_out = 48

                    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "wind")
                    dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_wind.X_test = X_test_seq

                    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "solar")
                    dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_solar.X_test = X_test_seq

                    features, targets, meteo_features, obs_features = return_dynamic_features(plant, "agr")
                    dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_agr.X_test = X_test_seq

                    dt_test_padding = dt_test_seq # we save the test datetimes for sequences for padding

                elif model == "LSTM":
                    n_lag = 144; n_out = 48

                    features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "wind")
                    dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_wind.X_test = X_test_LSTM

                    features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "solar")
                    dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_solar.X_test = X_test_LSTM

                    features, targets, meteo_features, obs_features  = return_dynamic_features(plant, "agr")
                    dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df_pertub, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
                    model_agr.X_test = X_test_LSTM

                    dt_test_padding = dt_test_LSTM # we save the test datetimes for sequences for padding

                # Make predictions # 
                Y_pred_wind = model_wind.test(filename = '')
                Y_pred_wind_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_wind, index = dt_test_padding)).values)
                
                Y_pred_solar = model_solar.test(filename = '')
                Y_pred_solar_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_solar, index = dt_test_padding)).values)
                
                Y_pred_agr = model_agr.test(filename = '')
                Y_pred_agr_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_agr, index = dt_test_padding)).values)
                
                Y_pred_agr_sum = (Y_pred_wind_padded*wind_cap + Y_pred_solar_padded*solar_cap)/agr_cap

                # We get the forecast errors
                Y_pred_agr_fc_error = Y_pred_agr_padded - Y_test
                Y_pred_agr_sum_fc_error = Y_pred_agr_sum - Y_test
                
                print("Saving simulations...")
                dict_wind[i] = Y_pred_wind_padded
                dict_solar[i] = Y_pred_solar_padded
                dict_agr[i] = Y_pred_agr_padded
                dict_agr_sum[i] = Y_pred_agr_sum
                dict_agr_fc_error[i] = Y_pred_agr_fc_error
                dict_agr_sum_fc_error[i] = Y_pred_agr_sum_fc_error

            # After simulating we look at the dataframe with simulations
            df_wind = pd.DataFrame(dict_wind, index = dt_test)
            df_solar = pd.DataFrame(dict_solar, index = dt_test)
            df_agr = pd.DataFrame(dict_agr, index = dt_test)
            df_agr_sum = pd.DataFrame(dict_agr_sum, index = dt_test)

            # We pad the data to the test datetimes
            df_index = pd.date_range(df_wind.index[0], df_wind.index[-1], freq = "30min")
            df_wind = pd.DataFrame(index = df_index).join(df_wind)

            df_index = pd.date_range(df_solar.index[0], df_solar.index[-1], freq = "30min")
            df_solar = pd.DataFrame(index = df_index).join(df_solar)   
            
            df_index = pd.date_range(df_agr.index[0], df_agr.index[-1], freq = "30min")
            df_agr = pd.DataFrame(index = df_index).join(df_agr)

            df_index = pd.date_range(df_agr_sum.index[0], df_agr_sum.index[-1], freq = "30min")
            df_agr_sum = pd.DataFrame(index = df_index).join(df_agr_sum)

            df_index = pd.date_range(df_wind.index[0], df_wind.index[-1], freq = "30min")
            df_wind = pd.DataFrame(index = df_index).join(df_wind)            

            # We add horizons
            horizons = np.tile(np.arange(48),int(len(df_agr.index)/48))
            df_wind['Horizon'] = horizons
            df_solar['Horizon'] = horizons
            df_agr['Horizon'] = horizons
            df_agr_sum['Horizon'] = horizons

            # We concat the simulations
            dict_vals_wind = {}
            dict_vals_solar = {}
            dict_vals_agr = {}
            dict_vals_agr_sum = {}
            for horizon in np.unique(horizons):
                # We drop the last column (horizon) and keep all the values
                dict_vals_wind[horizon] = np.concatenate(np.array(df_wind.loc[df_wind['Horizon'] == horizon])[:,:-1])
                dict_vals_solar[horizon] = np.concatenate(np.array(df_solar.loc[df_solar['Horizon'] == horizon])[:,:-1])
                dict_vals_agr[horizon] = np.concatenate(np.array(df_agr.loc[df_agr['Horizon'] == horizon])[:,:-1])
                dict_vals_agr_sum[horizon] = np.concatenate(np.array(df_agr_sum.loc[df_agr_sum['Horizon'] == horizon])[:,:-1])

            # We have a dataframe where each columns is a horizon with all simulated forecasted values
            df_vals_wind = pd.DataFrame(dict_vals_wind)
            df_vals_solar = pd.DataFrame(dict_vals_solar)
            df_vals_agr = pd.DataFrame(dict_vals_agr)
            df_vals_agr_sum = pd.DataFrame(dict_vals_agr_sum)

    # We save the mean and max std for wind / solar / agr
    max_std_list_wind.append(np.max(df_vals_wind.std()))
    mean_std_list_wind.append(np.mean(df_vals_wind.std()))

    max_std_list_solar.append(np.max(df_vals_solar.std()))
    mean_std_list_solar.append(np.mean(df_vals_solar.std()))

    max_std_list_agr.append(np.max(df_vals_agr.std()))
    mean_std_list_agr.append(np.mean(df_vals_agr.std()))

    max_std_list_agr_sum.append(np.max(df_vals_agr_sum.std()))
    mean_std_list_agr_sum.append(np.mean(df_vals_agr_sum.std()))

# We save a figure of the standard deviation per number of simulations
df_sims = pd.DataFrame({'max_std_wind': max_std_list_wind, 
                        'mean_std_wind': mean_std_list_wind,
                        'max_std_solar': max_std_list_solar,
                        'mean_std_solar': mean_std_list_solar,
                        'max_std_agr': max_std_list_agr,
                        'mean_std_agr': mean_std_list_agr,
                        'max_std_agr_sum': max_std_list_agr_sum,
                        'mean_std_agr_sum': mean_std_list_agr_sum}, 
                        index = n_sims)

df_sims.to_csv(f"{path_out}/df_simulation_check_{plant}_{model}_v4.csv", sep = ";")

fig_sims = df_sims.plot()
fig_sims.update_layout(template = 'ggplot2',
                                    yaxis_title = 'Standard Deviation',
                                    xaxis_title = 'Simulations (n)')
fig_sims.update_layout(title=go.layout.Title(text=f"Mean and maximum standard deviation across all horizons for different number of simulations",xref="paper",x=0))
fig_sims.write_image(f"{path_out}/simulation_check_{plant}_{model}_v4.png", width=1080, height=720)
print("Results exported.") 