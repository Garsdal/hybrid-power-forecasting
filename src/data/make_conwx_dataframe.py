import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import argparse
from src.utils import interp, setup_folders

def create_conwx_df(plant, solar_plant_ID, wind_plant_ID, meteo_ID, wind_plant_cap, solar_plant_cap):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed) for conwx format.
    """
    input_filepath = f'data/raw/{plant}'
    output_filepath = f'data/processed/{plant}'

    # Load obs
    path = f"data/raw/{plant}/Wind/{wind_plant_ID}.csv"
    df_obs_wind = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])

    path = f"data/raw/{plant}/Solar/{solar_plant_ID}.csv"
    df_obs_solar = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [0])

    df_merged = pd.concat([df_obs_wind, df_obs_solar], axis = 1)
    df_merged.columns = ['obs_power_wind', 'obs_power_solar']
    df_merged.index = pd.to_datetime(df_merged.index)

    # Load meteo
    path = f"data/raw/{plant}/Wind/{wind_plant_ID}_{meteo_ID}.csv"
    df_meteo_wind = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [1])
    df_meteo_wind = df_meteo_wind.loc[~df_meteo_wind.index.duplicated(keep='last')]

    path = f"data/raw/{plant}/Solar/{solar_plant_ID}_{meteo_ID}.csv"
    df_meteo_solar = pd.read_csv(path, sep = ";", parse_dates = True, index_col = [1])
    df_meteo_solar = df_meteo_solar.loc[~df_meteo_solar.index.duplicated(keep='last')]

    # We grab the frequency from observations
    res = df_merged.index.to_series().diff().value_counts().index[0]

    # We pad the data and interpolate meteo solar
    new_index = pd.date_range(df_meteo_solar.index[0], df_meteo_solar.index[-1], freq = res)
    df_meteo_solar = interp(df_meteo_solar, new_index)

    df = pd.concat([df_merged, df_meteo_wind, df_meteo_solar], axis = 1)
    df = df.dropna(subset = ['obs_power_wind', 'obs_power_solar'])

    # We create an aggregated column
    df['obs_power_agr'] = df['obs_power_solar'] + df['obs_power_wind']

    # We include a time of day feature
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    # We scale the power data with capacity before exporting
    agr_plant_cap = wind_plant_cap + solar_plant_cap
    df['obs_power_solar'] = df['obs_power_solar']/solar_plant_cap
    df['obs_power_wind'] = df['obs_power_wind']/wind_plant_cap
    df['obs_power_agr'] = df['obs_power_agr']/agr_plant_cap

    # We split the df into power and no_power
    df_power = df[["obs_power_solar", "obs_power_wind", "obs_power_agr"]]
    df_nopower = df.drop(["obs_power_solar", "obs_power_wind", "obs_power_agr"], axis=1)
 
    # We scale the remaining features
    Xscaler = MinMaxScaler(feature_range=(0, 1))
    Xscaler.fit(df_nopower)
    df_nopower.loc[:,:] =  Xscaler.transform(df_nopower)

    # We concat the scaled power and nopower
    df = pd.concat([df_power, df_nopower], axis = 1)
    
    # We create lagged power features
    df['obs_power_wind_lag1'] = df['obs_power_wind'].shift(1)
    df['obs_power_solar_lag1'] = df['obs_power_solar'].shift(1)
    df['obs_power_agr_lag1'] = df['obs_power_agr'].shift(1)

    # We find the resolution in minutes
    res_minutes = int(res.total_seconds()/60)

    # Export the data
    setup_folders(output_filepath)
    path_out = f'{output_filepath}/{plant}_OBS_METEO_{res_minutes}min.csv'
    df.to_csv(path_out, sep = ";", index = True)

    return 

# We allow the file to be run with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plant', type = str, default = 'HPP1')
parser.add_argument('--solar_ID', type = str)
parser.add_argument('--wind_ID', type = str)
parser.add_argument('--solar_cap', type = int)
parser.add_argument('--wind_cap', type = int)
parser.add_argument('--meteo_ID', type = str, default = "ECMWF")
args = parser.parse_args()

if __name__ == '__main__':
    plant = args.plant
    solar_ID = args.solar_ID
    wind_ID = args.wind_ID
    solar_cap = args.solar_cap
    wind_cap = args.wind_cap
    meteo_ID = args.meteo_ID
    create_conwx_df(plant, solar_ID, wind_ID, meteo_ID, wind_cap, solar_cap)