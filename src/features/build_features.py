# Takes as input a hybrid dataframe
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

### Input reshaping 
seed = 7
np.random.seed(seed) # fix the random seed for reproducible results in keras

def build_features(df, features, targets, dropna = True):
    if dropna:
        df = df.dropna()
    
    df['hour'] = np.round(df['hour']*23) # we remove the normalization for prettier dummy variables

    # We split the data into train/test
    df.loc[:,'weekday'] = df.index.day_name()
    train_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"]
    test_days = ["Friday"]
    val_days = ["Sunday"]

    training = df[df.isin(train_days).any(1)]
    test = df[df.isin(test_days).any(1)]
    val = df[df.isin(val_days).any(1)]

    # We grab our features and targets | we convert to numpy since we are no longer normalizing
    X_train = training[features]
    X_test = test[features]
    X_val = val[features]

    Y_train_scaled = np.array(training[targets])
    Y_test_scaled = np.array(test[targets])
    Y_val_scaled = np.array(val[targets])

    if 'hour' in features:
        X_train['hour'] = X_train['hour'].apply(str)
        X_test['hour'] = X_test['hour'].apply(str)
        X_val['hour'] = X_val['hour'].apply(str)
        X_train = pd.get_dummies(X_train, ['hour'])
        X_test = pd.get_dummies(X_test, ['hour'])
        X_val = pd.get_dummies(X_val, ['hour'])
    
    features_out = X_train.columns

    X_train_scaled = np.array(X_train)
    X_test_scaled = np.array(X_test)
    X_val_scaled = np.array(X_val)

    return(training.index, test.index, val.index, X_train_scaled, X_test_scaled, X_val_scaled, Y_train_scaled.ravel(), Y_test_scaled.ravel(), Y_val_scaled.ravel(), features_out)

def build_features_LSTM(df, obs_features, meteo_features, targets, n_lag = 144, n_out = 48, dropna = True):
    if dropna:
        df = df.dropna()
        
    # We define the targets out
    targets_out = targets

    # We grab our features and targets for observations and meteo seperately
    X_obs = df[obs_features]
    X_meteo = df[meteo_features]
    
    # Create sequences # 
    # ------------------------------
    # all historical data from SCADA
    X_past_obs_sorted = forecast_sequences_past(X_obs, n_lag)

    # ------------------------------
    # part historical, part future data from NIWE
    X_past_meteo = forecast_sequences_past(X_meteo, n_lag-n_out)

    # ------------------------------
    # Input sequences-from future observations
    X_future_meteo = forecast_sequences_future(X_meteo, n_out)

    # ------------------------------
    ### Concat input sequences past + future

    # If we dont sort here the columns will be [hour_t-96, DSWRF_t-96, WINDSPEED_t-96..., ]
    # After sorting they become [hour_t-96, hour_t-95,... DSWRF_t-96, ... DSWRF_t]

    # To obtain features in the shape [samples, feature, time] we concat piecewise and sort
    X_meteo_sorted = pd.concat([X_past_meteo, X_future_meteo], axis = 1).sort_index(axis = 1, ascending=False)
    # We concatenate the observations and meteo sorted
    df_X_seq = pd.concat([X_past_obs_sorted, X_meteo_sorted], axis=1)

    # We extract how the features have been sorted e.g something like [power_obs_agr, hour, .., DSWRF] since this will be the third dimension in our tensor
    features_out = pd.unique([x.split('(')[0] for x in df_X_seq.columns])

    # We split this X into train/test sequences
    df_X_seq.loc[:,'weekday'] = df_X_seq.index.day_name()
    train_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"]
    test_days = ["Friday"]
    val_days = ["Sunday"]

    X_train = df_X_seq[df_X_seq.isin(train_days).any(1)].drop(['weekday'],axis=1)
    X_test = df_X_seq[df_X_seq.isin(test_days).any(1)].drop(['weekday'],axis=1)
    X_val = df_X_seq[df_X_seq.isin(val_days).any(1)].drop(['weekday'],axis=1)

    # ------------------------------
    ### Output sequences - only future
    df_Y_seq = X_obs[targets]
    df_Y_seq = pd.DataFrame(df_Y_seq, index=X_obs.index)
    df_Y_seq = forecast_sequences_future(df_Y_seq, n_out)

    # We split this Y into train/test sequences
    df_Y_seq.loc[:,'weekday'] = df_Y_seq.index.day_name()
    Y_train = df_Y_seq[df_Y_seq.isin(train_days).any(1)].drop(['weekday'],axis=1)
    Y_test = df_Y_seq[df_Y_seq.isin(test_days).any(1)].drop(['weekday'],axis=1)
    Y_val = df_Y_seq[df_Y_seq.isin(val_days).any(1)].drop(['weekday'],axis=1)

    # ------------------------------
    ### We only keep the complete Y-sequences and corresponding inputs
    #  Removing rows where outputs are NaNs 
    idx = (np.isnan(Y_train).any(axis=1))
    X_train = X_train.loc[~idx, :]
    Y_train = Y_train.loc[~idx, :]

    idx = (np.isnan(Y_test).any(axis=1))
    X_test= X_test.loc[~idx, :]
    Y_test = Y_test.loc[~idx, :]

    idx = (np.isnan(Y_val).any(axis=1))
    X_val= X_val.loc[~idx, :]
    Y_val = Y_val.loc[~idx, :]

    # We convert to numpy arrays and rename for the rest of the code
    X_train_scaled = np.array(X_train)
    X_test_scaled = np.array(X_test)
    X_val_scaled = np.array(X_val)
    Y_train_scaled = np.array(Y_train)
    Y_test_scaled = np.array(Y_test)
    Y_val_scaled = np.array(Y_val)

    # MASKING MISSING DATA
    pad_value = 999

    # input padding
    X_train_scaled[np.isnan(X_train_scaled)] = pad_value
    X_test_scaled[np.isnan(X_test_scaled)] = pad_value
    X_val_scaled[np.isnan(X_val_scaled)] = pad_value

    # ------------------------------
    ### Input reshape for LSTM problem 
    no_features = int(X_train_scaled.shape[1]/n_lag)

    #[samples, features, timesteps] the above reshape without sorting meteo will not have features in the 3rd dimension
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], no_features, n_lag))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], no_features, n_lag))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], no_features, n_lag))

    #[samples, timesteps, features] After doing the correct reshape we move the features to the 'last' dimension
    X_train_scaled = np.transpose(X_train_scaled, (0, 2, 1))
    X_test_scaled = np.transpose(X_test_scaled, (0, 2, 1))
    X_val_scaled = np.transpose(X_val_scaled, (0, 2, 1))

    return(X_train.index, X_test.index, X_val.index, X_train_scaled, X_test_scaled, X_val_scaled, Y_train_scaled, Y_test_scaled, Y_val_scaled, features_out, targets_out)

def build_features_seq(df, obs_features, meteo_features, targets, n_lag = 144, n_out = 48, dropna = True):
    if dropna:
        df = df.dropna()

    # We define the features
    targets_out = targets

    # We grab our features and targets for observations and meteo seperately
    X_obs = df[obs_features]
    X_meteo = df[meteo_features]
    
    # Create sequences # 
    # ------------------------------
    # all historical data from SCADA
    X_past_obs_sorted = forecast_sequences_past(X_obs, n_lag)

    # ------------------------------
    # part historical, part future data from NIWE
    X_past_meteo = forecast_sequences_past(X_meteo, n_lag-n_out)

    # ------------------------------
    # Input sequences-from future observations
    X_future_meteo = forecast_sequences_future(X_meteo, n_out)

    # ------------------------------
    ### Concat input sequences past + future

    # To obtain features in the shape [samples, feature, time] we concat piecewise and sort
    X_meteo_sorted = pd.concat([X_past_meteo, X_future_meteo], axis = 1).sort_index(axis = 1, ascending=False)

    # We create a dataframe with the created sequences
    df_X_seq = pd.concat([X_past_obs_sorted, X_meteo_sorted], axis=1)
    
    # We split this X into train/test sequences
    df_X_seq.loc[:,'weekday'] = df_X_seq.index.day_name()
    train_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"]
    test_days = ["Friday"]
    val_days = ["Sunday"]

    X_train = df_X_seq[df_X_seq.isin(train_days).any(1)].drop(['weekday'],axis=1)
    X_test = df_X_seq[df_X_seq.isin(test_days).any(1)].drop(['weekday'],axis=1)
    X_val = df_X_seq[df_X_seq.isin(val_days).any(1)].drop(['weekday'],axis=1)

    # ------------------------------
    ### Output sequences - only future
    df_Y_seq = X_obs[targets]
    df_Y_seq = pd.DataFrame(df_Y_seq, index=X_obs.index)
    df_Y_seq = forecast_sequences_future(df_Y_seq, n_out)

    # We split this Y into train/test sequences
    df_Y_seq.loc[:,'weekday'] = df_Y_seq.index.day_name()
    Y_train = df_Y_seq[df_Y_seq.isin(train_days).any(1)].drop(['weekday'],axis=1)
    Y_test = df_Y_seq[df_Y_seq.isin(test_days).any(1)].drop(['weekday'],axis=1)
    Y_val = df_Y_seq[df_Y_seq.isin(val_days).any(1)].drop(['weekday'],axis=1)

    # ------------------------------
    ### We only keep the complete Y-sequences and corresponding inputs
    #  Removing rows where outputs are NaNs 
    idx = (np.isnan(Y_train).any(axis=1))
    X_train = X_train.loc[~idx, :]
    Y_train = Y_train.loc[~idx, :]

    idx = (np.isnan(Y_test).any(axis=1))
    X_test= X_test.loc[~idx, :]
    Y_test = Y_test.loc[~idx, :]

    idx = (np.isnan(Y_val).any(axis=1))
    X_val= X_val.loc[~idx, :]
    Y_val = Y_val.loc[~idx, :]

    # We have to drop all nan rows for X as well since we can't pad with 999 for sklearn
    idx = (np.isnan(X_train).any(axis=1))
    X_train = X_train.loc[~idx, :]
    Y_train = Y_train.loc[~idx, :]

    idx = (np.isnan(X_test).any(axis=1))
    X_test= X_test.loc[~idx, :]
    Y_test = Y_test.loc[~idx, :]

    idx = (np.isnan(X_val).any(axis=1))
    X_val= X_val.loc[~idx, :]
    Y_val = Y_val.loc[~idx, :]

    # Features and targets
    features_out = X_train.columns
    targets_out = Y_train.columns
    dt_train = X_train.index
    dt_test = X_test.index
    dt_val = X_val.index

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_val = np.array(Y_val)

    return(dt_train, dt_test, dt_val, X_train, X_test, X_val, Y_train, Y_test, Y_val, features_out, targets_out)


#%% Functions for sequential transformation used in build_features_LSTM()
def forecast_sequences_past(past_data,n_lag):
    """
    A function that will split the 'past' time series to sequences for nowcast/forecast problems
    Arguments:
        past_data: Time series of 'past' observations as a list, NumPy array or pandas series
        n_lag: number of previous time steps to include within the sequence, a.k.a. time-lag        
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = past_data.shape[1] 
    df = pd.DataFrame(past_data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_lag, 0, -1):
        cols.append(df.shift(i))
        #names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        names += [f'{df.columns[j]}(t-{i})' for j in range(n_vars)]
    # put it all together (aggregate)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


### define a function that will prepare the shifting future sequences of the network
def forecast_sequences_future(future_data,n_out):
    """
    A function that will split the 'future' time series to sequences for nowcast/forecast problems
    Arguments:
        future_data: Time series of future observations as a list, NumPy array or pandas series
        n_out: number forecast time steps within the horizon (for multi-output forecast)
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = future_data.shape[1] 
    df = pd.DataFrame(future_data)
    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n-1)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            #names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            names += [f'{df.columns[j]}(t)' for j in range(n_vars)]
        else:
            #names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            names += [f'{df.columns[j]}(t+{i})' for j in range(n_vars)]
    # put it all together (aggregate)
    agg = pd.concat(cols, axis=1)
    agg.columns = names    
    return agg
