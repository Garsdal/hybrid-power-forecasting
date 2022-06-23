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
parser.add_argument('--bootstrap', type = bool, default = True)
parser.add_argument('--max_depth', type = int, default = None)
parser.add_argument('--min_samples_leaf', type = int, default = 1)
parser.add_argument('--min_samples_split', type = int, default = 2)
parser.add_argument('--n_estimators', type = int, default = 100)
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
params = {'max_depth': args.max_depth,
            'min_samples_leaf': args.min_samples_leaf,
            'min_samples_split': args.min_samples_split,
            'n_estimators': args.n_estimators}

# Create fullday features for RF
n_lag = 144; n_out = 48
features, targets, meteo_features, obs_features  = return_dynamic_features(plant, tech)
dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq = build_features_seq(df, obs_features, meteo_features, targets, n_lag = n_lag, n_out = n_out)
print(dt_train.shape, dt_test.shape, dt_val.shape, X_train.shape, X_test.shape, X_val.shape, Y_train.shape, Y_test.shape, Y_val.shape)
print(dt_train_seq.shape, dt_test_seq.shape, dt_val_seq.shape, X_train_seq.shape, X_test_seq.shape, X_val_seq.shape, Y_train_seq.shape, Y_test_seq.shape, Y_val_seq.shape)

# We train a sequential RF model
static, seq = False, True
RF_model_seq = RF(dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq, static = static, seq = seq, runtime = runtime, wandb = wandb, **params)
RF_model_seq.train(plant)

# We load a trained sequential model
static_bool = 'static' if static else 'dynamic'
filename = f"{root_location}/{scratch_location}/models_tuning/RF/{plant}/RF_{plant}_{static_bool}_numf{len(features_seq)}_{targets_seq[0]}_{runtime}.joblib"
RF_model_seq = RF(dt_train_seq, dt_test_seq, dt_val_seq, X_train_seq, X_test_seq, X_val_seq, Y_train_seq, Y_test_seq, Y_val_seq, features_seq, targets_seq, static = static, seq = seq)

# We make predictions for the test and validation set | here we pad the predictions to the non-sequence datetimes
Y_pred_RF_test = RF_model_seq.test(filename = filename, set = "test")
Y_pred_RF_test_padded = np.concatenate(pd.DataFrame(index = dt_test).join(pd.DataFrame(Y_pred_RF_test, index = dt_test_seq)).values)

Y_pred_RF_val = RF_model_seq.test(filename = filename, set = "val")
Y_pred_RF_val_padded = np.concatenate(pd.DataFrame(index = dt_val).join(pd.DataFrame(Y_pred_RF_val, index = dt_val_seq)).values)

# We calculate the test error | drop nans first
idx = np.isnan(Y_pred_RF_test_padded)
MSE_test = mean_squared_error(Y_test[~idx], Y_pred_RF_test_padded[~idx])
wandb.log({'MSE_test': MSE_test})

idx = np.isnan(Y_pred_RF_val_padded)
MSE_val = mean_squared_error(Y_val[~idx], Y_pred_RF_val_padded[~idx])
wandb.log({'MSE_val': MSE_val})