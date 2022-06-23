import os
import wandb
import argparse
from datetime import datetime as dt

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import json

# have to run pip install -e . before running
from src.utils import setup_folders, return_static_features, return_dynamic_features
from src.features.build_features import build_features, build_features_LSTM, build_features_seq
from src.models.deterministic.models import RF, LR, LGB, Persistence, my_LSTM

# Selections
plant = "HPP1"

# We grab the runtime for unique model names
runtime = dt.now().strftime("%m%d_%H%M_%S")

# Team API key
f = open('src/sweeps/wandb_key.json')
wandb_key = json.load(f)
f.close()
os.environ["WANDB_API_KEY"] = wandb_key["key"]

# We allow the file to be run with argparse for a wandb sweep
parser = argparse.ArgumentParser()
parser.add_argument('--plant', type = str, default = "HPP1")
parser.add_argument('--tech', type = str, default = "agr")
parser.add_argument('--activation', type = str, default = "tanh")
parser.add_argument('--n_lag', type = int, default = 144)
parser.add_argument('--epochs', type = int, default = 2)
parser.add_argument('--batch_size', type = int, default = 1000)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--neurons_l1', type = int, default = 100)
parser.add_argument('--neurons_l2', type = int, default = 50)
parser.add_argument('--neurons_l3', type = int, default = 25)
args = parser.parse_args()

# Init W&B
wandb.init(project="hybrid-power-forecasting", entity="garsdal")

# Get the plant
plant = args.plant

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

tech = args.tech
wandb.log({'tech': tech})

# For scratch model loading
root_location=os.path.abspath(os.sep)
scratch_location='work3/s174440'

# Get hyperparams from argparse
params = {'activation': args.activation,
            'n_lag': args.n_lag,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'neurons_l1': args.neurons_l1,
            'neurons_l2': args.neurons_l2,
            'neurons_l3': args.neurons_l3}

### We specify the features and targets
features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)

### Create full sequence LSTM features
n_lag = args.n_lag; n_out = 48
dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM = build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)

### Full sequence LSTM
epochs = args.epochs
LSTM_model = my_LSTM(dt_train_LSTM, dt_test_LSTM, dt_val_LSTM, X_train_LSTM, X_test_LSTM, X_val_LSTM, Y_train_LSTM, Y_test_LSTM, Y_val_LSTM, features_LSTM, targets_LSTM, recursive = False, runtime = runtime, wandb = wandb, **params)
LSTM_model.train(plant, epochs = epochs, batch_size = args.batch_size)

# We make predictions for full sequence
filename = f"{root_location}/{scratch_location}/models_tuning/LSTM/{plant}/LSTM_nlag{n_lag}_nout{n_out}_epochs{epochs}_{plant}_numf{len(features_LSTM)}_{targets_LSTM[0]}_{runtime}.h5"

Y_pred_LSTM_test = LSTM_model.test(filename = filename, set = "test")
Y_pred_LSTM_test_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_LSTM_test, index = dt_test_LSTM)).values)

Y_pred_LSTM_val = LSTM_model.test(filename = filename, set = "val")
Y_pred_LSTM_val_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_LSTM_val, index = dt_val_LSTM)).values)

# We calculate the test error | drop nans first
idx = np.isnan(Y_pred_LSTM_test_padded)
MSE_test = mean_squared_error(Y_test[~idx], Y_pred_LSTM_test_padded[~idx])
wandb.log({'MSE_test': MSE_test})

idx = np.isnan(Y_pred_LSTM_val_padded)
MSE_val = mean_squared_error(Y_val[~idx], Y_pred_LSTM_val_padded[~idx])
wandb.log({'MSE_val': MSE_val})