# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math

from sklearn.preprocessing import MinMaxScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    plant = input_filepath.split("/")[2]
    wind_plant_cap = 100#mw
    solar_plant_cap = 100#mw

    # SOLAR #
    solar_path = f'{input_filepath}/Solar/SCADA_ATM_FC_30min_Solar.csv'
    df_SOLAR = pd.read_csv(solar_path, sep = ";", parse_dates = True, index_col = [0])

    # WIND #
    wind_path = f'{input_filepath}/Wind/SCADA_ATM_FC_30min_Wind.csv'
    df_WIND = pd.read_csv(wind_path, sep = ";", parse_dates = True, index_col = [0])

    # Create Hybrid #
    df = pd.concat([df_SOLAR, df_WIND], axis = 1)
    df = df.loc[:,~df.columns.duplicated()]

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

    # Create lagged power features
    df['obs_power_wind_lag1'] = df['obs_power_wind'].shift(1)
    df['obs_power_solar_lag1'] = df['obs_power_solar'].shift(1)
    df['obs_power_agr_lag1'] = df['obs_power_agr'].shift(1)

    hybrid_path = f'{output_filepath}/{plant}_OBS_METEO_30min.csv'
    df.to_csv(hybrid_path, sep = ";", index = True)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()