import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
import json

from scipy.stats import norm, gamma
import random

import logging
import logging.config

def setup_folders(folder):
        if not os.path.exists(folder):
            print(f"{folder} does not exist... creating")
            os.makedirs(folder)

def shift1(arr, num, fill_value=np.nan):
    arr = np.roll(arr,num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr

# We interpolate solar
def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)

def init_logging(logger, model, plant, features, targets, X_train, X_test, X_val, Y_train, Y_test, Y_val, params, static_bool = None, recursive_bool = None):
    logging.info(f"Starting training of {model} model for plant {plant}")
    logging.info(f"Features: {features}")
    logging.info(f"Targets: {targets}")
    logging.info(f"The run is static/dynamic: {static_bool}")
    logging.info(f"The run is recursive: {recursive_bool}")
    logging.info(f"Shape of X_train: {X_train.shape}")
    logging.info(f"Shape of X_test: {X_test.shape}")
    logging.info(f"Shape of X_val: {X_val.shape}")
    logging.info(f"Shape of Y_train: {Y_train.shape}")
    logging.info(f"Shape of Y_test: {Y_test.shape}")
    logging.info(f"Shape of Y_val: {Y_val.shape}")
    if len(params) != 0:
        logging.info(json.dumps(params, indent = 4))
    else:
        logging.info("Model default hyperparameters used.")
    return(logger)

def return_dynamic_features(plant, tech):
    # The target
    targets = [f'obs_power_{tech}']

    ### We specify the features and targets
    if plant == "Nazeerabad":
        if tech == "wind":
            features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_wdir_101.8m_4km_NIWE_NEW', f'obs_power_wind_lag1', 
            'hour'] 
            meteo_features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_wdir_101.8m_4km_NIWE_NEW', 
            'hour']
        elif tech == "solar":
            features = ['fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE', 'obs_power_solar_lag1',
            'hour'] 
            meteo_features = ['fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE',
            'hour']
        elif tech == "agr":
            features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_wdir_101.8m_4km_NIWE_NEW',
                            'fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE',
                            'obs_power_agr_lag1',
                            'hour'] 
            meteo_features = ['fc_ws_101.8m_4km_NIWE', 'fc_wdir_101.8m_4km_NIWE',
                            'fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE', 
                            'hour']
        obs_features = [f'obs_power_{tech}']
    elif plant in ["HPP1", "HPP2", "HPP3"]:
        if tech == "wind":
            features = ['WINDSPEED_100m', 'WINDDIR_100m', 'obs_power_wind_lag1',
            'hour']
            meteo_features = ['WINDSPEED_100m', 'WINDDIR_100m',
            'hour']
        elif tech == "solar":
            features = ['DSWRF', 'Tmp_2m_C', f'obs_power_solar_lag1',
            'hour']
            meteo_features = ['DSWRF', 'Tmp_2m_C',
            'hour']
        elif tech == "agr":
            features = ['WINDSPEED_100m', 'WINDDIR_100m',
                        'DSWRF', 'Tmp_2m_C',
                        'obs_power_agr_lag1',
                        'hour']
            meteo_features = ['WINDSPEED_100m', 'WINDDIR_100m',
                            'DSWRF', 'Tmp_2m_C', 
                            'hour']
        obs_features = [f'obs_power_{tech}']
    
    return(features, targets, meteo_features, obs_features)

def return_static_features(plant, tech):
    if plant == "Nazeerabad":
        targets = [f'obs_power_{tech}']
        if tech == "wind":
            features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_wdir_101.8m_4km_NIWE_NEW',
            'hour'] 
        elif tech == "solar":
            features = ['fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE',
            'hour']
        elif tech == "agr":
            features = ['fc_ws_101.8m_4km_NIWE_NEW', 'fc_wdir_101.8m_4km_NIWE_NEW', 
                        'fc_ghi_4km_NIWE', 'fc_temp_4km_NIWE', 
                        'hour']
    elif plant in ["HPP1", "HPP2", "HPP3"]:
        targets = [f'obs_power_{tech}']
        if tech == "wind":
            features = ['WINDSPEED_100m', 'WINDDIR_100m',
            'hour']
        elif tech == "solar":
            features = ['DSWRF', 'Tmp_2m_C', 
            'hour']
        elif tech == "agr":
            features = ['WINDSPEED_100m', 'WINDDIR_100m',
                    'DSWRF', 'Tmp_2m_C',
                    'hour']
    
    return(features, targets)

# Function for copula coupled uniform variables
def gen_gauss_copula(rho_s, n, seed):
    np.random.seed(seed = seed)
    
    rho = 2*np.sin(rho_s * np.pi/6) # pearson correlation
    
    P = np.array([[1, rho],
                 [rho, 1]]) # correlation matrix 
    
    d = P.shape[0]
    
    #Z = np.random.normal(size = (n,d)) # random normal variables
    #Z = np.dot(Z, np.transpose(np.linalg.cholesky(P))) # cholesky factorization for correlation
    Z = np.random.multivariate_normal([0,0], P, n)

    # We apply the marginal distributions (uniform dist is the cdf) 
    U = norm.cdf(Z)
    
    # We try different marginal functions for X
    X = U
    X[:,0] = gamma.ppf(U[:,0], a = 5)
    X[:,0] = X[:,0]-np.mean(X[:,0])
    X[:,1] = Z[:,1]
    
    X = pd.DataFrame(X)
    X = (X-X.mean()) / X.std() # standardization
    X = X/X.max().values # normalization

    return(pd.DataFrame(X)) # return gamma and normal dist copula | corrected to be in [-1, 1]
    #return(pd.DataFrame(Z)) # we return the correlated normal copula
    #return(pd.DataFrame(U) - 0.5) # we return the uniform variable copula

def create_weights(max_uncertainty_wind = 0.1, max_uncertainty_solar = 0.1):
    sigma = 3 
    # the width of the gaussian kernel was adjusted by trial and error to improve the fit of std for all horizons
    # with too wide a kernel we would add too much noise in the horizons in the morning/evening and too narrow not enough.

    weights_wind = np.roll(np.linspace(max_uncertainty_wind,max_uncertainty_wind,48), 12)
    
    # we control the uncertainty at the middle of the day
    weights_solar_gauss = 1/(np.sqrt(2*np.pi)*sigma)*np.exp((-np.linspace(-1,1,48)**2)/2*sigma**2)
    weights_solar_gauss_norm = (weights_solar_gauss / np.max(weights_solar_gauss))*max_uncertainty_solar 

    weights = pd.DataFrame({'wind_weight': weights_wind, 'solar_weight': weights_solar_gauss_norm})

    return weights

# Function to create perturbed inputs
def create_pertubed_inputs(df, meteo_features, weights, seed, return_corr = False):
    np.random.seed(seed = seed)
    random.seed(seed)
    
    df_pertub = df.copy(deep = True)
    
    # We pad and add a horizon
    df_index = pd.date_range(df_pertub.index[0], df_pertub.index[-1], freq = "30min")
    df_pertub = pd.DataFrame(index = df_index).join(df_pertub)
    horizons = np.tile(np.arange(48),int(len(df_pertub.index)/48)+1)
    df_pertub['Horizon'] = horizons[:df_pertub.shape[0]]
    
    corrs_before = []
    corrs_after = []
    std_before = []
    std_after = []
    mean_before = []
    mean_after = []
    # We add uniform noise for each horizon
    for horizon in np.unique(df_pertub['Horizon']):
        # We set a seed so that we can reproduce the results
        np.random.seed(seed = horizon)
        
        # We get the index of the horizons 
        idx = df_pertub['Horizon'] == horizon
        
        # Horizon values and correlation between wind speed + GHI for this horizon
        horizon_values = df_pertub.loc[idx, meteo_features]
        corr_horizon = np.nan_to_num(np.array(horizon_values.corr(method = 'spearman')))[0,1]
        
        # We save the corr/mean/std before
        corrs_before.append(corr_horizon)
        std_before.append(np.std(horizon_values[meteo_features[0]]))
        mean_before.append(np.mean(horizon_values[meteo_features[0]]))
        
        # uniform values in the range [-1,1] with correlation scaled by weights
        noise_samples = gen_gauss_copula(corr_horizon, horizon_values.shape[0], seed = horizon*seed)*weights.loc[horizon].values

        horizon_values_pertubed = np.array(horizon_values) + np.array(noise_samples)
        horizon_values_pertubed = pd.DataFrame(horizon_values_pertubed, index = horizon_values.index)
        
        # We set all negative values to 0
        horizon_values_pertubed[horizon_values_pertubed < 0] = 0 

        # We add the values with noise
        df_pertub.loc[idx, meteo_features] = horizon_values_pertubed.values
    
        # We save the corr/mean/std after
        corrs_after.append(np.array(horizon_values_pertubed.corr(method = "spearman"))[0,1])
        std_after.append(np.max(np.std(horizon_values_pertubed)))
        mean_after.append(np.mean(horizon_values_pertubed[0]))

    df_corrs = pd.DataFrame({'corr_pre': corrs_before, 'corr_post': corrs_after}, index = np.arange(48))
    df_std = pd.DataFrame({'std_pre': std_before, 'std_post': std_after}, index = np.arange(48))
    df_mean = pd.DataFrame({'mean_pre': mean_before, 'mean_post': mean_after}, index = np.arange(48))

    # We add 1% uniform error to all power observations
    power_cols = ['obs_power_wind', 'obs_power_solar', 'obs_power_agr']
    for col in power_cols:
        power_values_pertubed = df_pertub[col].values + np.random.uniform(low = -0.5, high = 0.5, size = df_pertub[col].shape)/100
        df_pertub[col] = power_values_pertubed

    if return_corr:
        return(df_pertub, df_corrs, df_std, df_mean)
    else:
        return(df_pertub)

# used for uncertainty propagation
def concat_day_simulations(df_agr, df_agr_sum, day, horizons):
    dict_vals_agr = {}
    dict_vals_agr_sum = {}

    for horizon in np.unique(horizons):
        # We drop the last columns (Y_true, horizon) and keep all the values
        dict_vals_agr[horizon] = np.concatenate(np.array(df_agr[day].loc[df_agr['Horizon'] == horizon])[:,:-2])
        dict_vals_agr_sum[horizon] = np.concatenate(np.array(df_agr_sum[day].loc[df_agr_sum['Horizon'] == horizon])[:,:-2])

    df_vals_agr_day = pd.DataFrame(dict_vals_agr)
    df_vals_agr_sum_day = pd.DataFrame(dict_vals_agr_sum)
    
    return(df_vals_agr_day, df_vals_agr_sum_day)

# used for uncertainty propagation
def create_quantiles(df_agr, df_vals_agr):
    #dict_quantiles = {0.01: np.array([]),
    #                      0.05: np.array([]),
    #                      0.1: np.array([]), 
    #                      0.25: np.array([]), 
    #                      0.4: np.array([]),
    #                      0.6: np.array([]),
    #                      0.75: np.array([]), 
    #                      0.9: np.array([]),
    #                      0.95: np.array([]),
    #                      0.99: np.array([])}

    dict_quantiles = {0.05: np.array([]),
                          0.1: np.array([]), 
                          0.25: np.array([]), 
                          0.75: np.array([]), 
                          0.9: np.array([]),
                          0.95: np.array([]),}

    #dict_quantiles = {0.01: np.array([]),
    #                      0.05: np.array([]),
    #                      0.1: np.array([]), 
    #                      0.9: np.array([]),
    #                      0.95: np.array([]),
    #                      0.99: np.array([])}

    #dict_quantiles = {0.05: np.array([]),
    #                  0.95: np.array([])}

    for horizon in df_vals_agr.columns:
        forecast_values_horizon = df_vals_agr[horizon].dropna()
         # We grab the quantiles
        for quantile in dict_quantiles.keys():
            dict_quantiles[quantile] = np.append(dict_quantiles[quantile], np.quantile(forecast_values_horizon, quantile))

    df_quantiles = pd.DataFrame(dict_quantiles, index = np.unique(df_agr['Horizon'])) 
    
    return(df_quantiles)