import numpy as np
import pandas as pd
import datetime

### Define forecasting metrics ### Credit to DTU Wind

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    """ NOTE: not normalized by mean to avoid 0-mean observations """
    return rmse(actual, predicted) / (actual.max() - actual.min())
    #return rmse(actual, predicted) / actual.mean()

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """ Percentage error """
    #return _error(actual, predicted) / actual
    return _error(actual, predicted) / (actual.max() - actual.min())

def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.nanmean(_percentage_error(actual, predicted))

def std(actual: np.ndarray, predicted: np.ndarray):
    """ Standard deviation """
    return np.std(_error(actual, predicted))

# Function for all horizons
def calculate_metric_horizons(df, col_true, col_model, method):
    methods = {'me': me,
               'mae': mae,
               'mse': mse,
               'mpe': mpe,
               'rmse': rmse,
               'nrmse': nrmse,
               'std': std}

    # Requires that the df has a 'horizon' column to run this
    horizons = np.unique(df['Horizon'])

    # We have to drop nans for the methods to work
    df = df.dropna()

    metrics = []
    n_horizons = []
    for horizon in horizons:

        Y_true = df.loc[df['Horizon'] == horizon,col_true].values
        Y_pred = df.loc[df['Horizon'] == horizon,col_model].values
        
        # We calculate the metric values
        metric = methods[method](Y_true, Y_pred)
        metrics.append(metric)
        n_horizons.append(len(Y_true))

    return(metrics)

# We create a calculate metric function for different horizon windows (how does the model perform )
def calculate_metric_horizon_windows(df, method, horizons_windows):
    methods = {'me': me,
               'mae': mae,
               'mse': mse,
               'mpe': mpe,
               'rmse': rmse,
               'nrmse': nrmse}

    # We select the specified 
    method_func = methods[method]

    # We drop the padded datetimes
    df = df.dropna()

    # We grab the Y_true
    Y_true = df['Y_true'].values
    Horizons = df['Horizon']
    df = df.drop(columns = ['Y_true', 'Horizon'])

    stats = {}
    for window in horizons_windows:

         # We loop over each model prediction
        metrics = []
        for col in df.columns:
            Y_pred = df[col].values
            
            idx_bool = Horizons < window
            Y_pred_window = Y_pred[idx_bool]
            Y_true_window = Y_true[idx_bool]

            # We calculate for the given metric and add to a list
            metrics.append(method_func(Y_true_window, Y_pred_window))

            # We append all the metrics to the dict with a key for the window
            stats[window] = metrics

    df_metrics = pd.DataFrame(stats, index = df.columns)

    return(df_metrics)

def calculate_metric_horizons_all_models(df, method):
     # We grab all columns in the df_pred except and 'Horizon'
    models = df.columns.drop(['Horizon', 'Y_true'])

    # We create a template dataframe for the RMSE for each horizon for each model
    df_metrics = pd.DataFrame(columns=models)
    for model in models:
        df_metrics[model] = calculate_metric_horizons(df,'Y_true', model, method)

    return(df_metrics)