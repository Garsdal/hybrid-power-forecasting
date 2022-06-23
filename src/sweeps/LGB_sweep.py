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
parser.add_argument('--max_depth', type = int, default = None)
parser.add_argument('--feature_fraction', type = float, default = 1)
parser.add_argument('--num_leaves', type = int, default = 31)
parser.add_argument('--num_iterations', type = int, default = 100)
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
            'feature_fraction': args.feature_fraction,
            'num_leaves': args.num_leaves,
            'num_iterations': args.num_iterations}

### We make static features which contain no power information
features, targets = return_static_features(plant, tech)
dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features = build_features(df, features, targets)
print(dt_train.shape, dt_test.shape, dt_val.shape, X_train.shape, X_test.shape, X_val.shape, Y_train.shape, Y_test.shape, Y_val.shape)

### We train the LGB
static = True
LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static, runtime = runtime, wandb = wandb, **params)
LGB_model.train(plant)

### We load a trained model
static_bool = 'static' if static else 'dynamic'
filename = f"{root_location}/{scratch_location}/models_tuning/LGB/{plant}/LGB_{plant}_{static_bool}_numf{len(features)}_{targets[0]}_{runtime}.sav"
LGB_model = LGB(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features, targets, static = static)

# We make predictions for the test and validation set 
Y_pred_LGB_test = LGB_model.test(filename = filename, set = "test")
Y_pred_LGB_val = LGB_model.test(filename = filename, set = "val")

# We calculate the error | drop nans first
MSE_test = mean_squared_error(Y_test, Y_pred_LGB_test)
wandb.log({'MSE_test': MSE_test})

MSE_val = mean_squared_error(Y_val, Y_pred_LGB_val)
wandb.log({'MSE_val': MSE_val})