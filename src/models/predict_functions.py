from re import L
import numpy as np
import pandas as pd
from pyparsing import java_style_comment
import pickle
import datetime
from src.utils import setup_folders, shift1, init_logging, LoggingCallback

################ Regression Prediction Functions ###############

def predict_step_ahead(model, X_test, timestep, col, freq):
    # We grab a sample for a given dt
    X_sample = X_test.loc[timestep,:].values.reshape(1,-1)

    # We make a prediction
    Y_pred_recursive = model.predict(X_sample)
    
    # We set the predicted power equal to the lagged power in the next step | This makes the forecast recursive
    next_step = timestep + freq
    
    # If we can find the next step in X_test datetimes we update the value | This will make sure that when we get to the end of a test day then we don't update any incorrect values
    if next_step in X_test.index:
        X_test.loc[next_step, col] = Y_pred_recursive
    else:
        pass

    return(Y_pred_recursive, X_test)

def recursive_predict_test_days(model, X_test, dt_test, features, targets):
    # We convert X_test to a dataframe to use datetimes for other models
    X_test = pd.DataFrame(X_test, index = dt_test, columns = features)

    # We get forecast days
    forecast_days = dt_test[dt_test.time == datetime.time(0, 0)]

    # We get the col_name of power_obs_lag1 based on the target
    col = "_".join([targets[0], "lag1"])
    print("Features:", features)
    print("Updating recursively:", col)
    # We also get the frequency
    freq = X_test.index.to_series().diff().value_counts().index[0]

    # We create the recursive forecast for all forecasts days
    dt_list = []
    val_list = []
    print("Predicting recursively for test days...")
    for i, day in enumerate(forecast_days):
        #print("Day", i+1, "out of", len(forecast_days))

        # We get a dynamic dt_range (allows for test days with nans) which allows test days to not have full length
        dt_range = X_test.loc[day.strftime('%Y-%m-%d')].index
        dt_list.append(dt_range)

        # We create the recursive forecast using the predict_step_ahead function
        predictions = []
        for timestep in dt_range:
            # We make a power prediction and update recursively for a given datetime
            Y_pred_recursive, X_test = predict_step_ahead(model, X_test, timestep, col, freq)

            # We save the prediction
            predictions.append(Y_pred_recursive)

        val_list.append(predictions)

    dt_list = np.concatenate(np.array(dt_list)).reshape(-1)
    val_list = np.concatenate(np.array(val_list)).reshape(-1)

    # For now we match to the test set
    df_recursive = pd.DataFrame({'power_pred': val_list}, index = dt_list)

    # For now we join the df_persistence on the dt_test | we might be missing values for some days
    df_index = pd.DataFrame(index = dt_test)
    df_merged = df_index.join(df_recursive)
    
    # We get the predictions
    Y_pred = np.concatenate(df_merged.values)

    return(Y_pred)

############# LSTM Prediction Functions ##################

def predict_full_sequence_days(model, X_test, dt_test):
    # We make predictions for the whole test set and extract only the sequences which cover our test days
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, index=dt_test)

    # We grab predictions for each day in the test set (this has to be the LSTM days where we have dropped non complete sequences)
    forecast_days = dt_test[dt_test.time == datetime.time(0, 0)]

    # We append the predicted full day sequences for all forecast days
    val_list = []
    dt_list = []
    print("Predicting full day sequences for test days...")
    for i, day in enumerate(forecast_days):
        #print("Day", i+1, "out of", len(forecast_days))

        # We get a full day dt_range since we have dropped sequences with NaNs for a full day (incomplete Y-sequences dropped)
        dt_range = pd.date_range(day, day+datetime.timedelta(days=1,minutes=-30), freq = "30min")
        dt_list.append(dt_range)

        # We grab the predictions for the specific forecast day
        val_list.append(Y_pred.loc[day].values)

    dt_list = np.concatenate(np.array(dt_list)).reshape(-1)
    val_list = np.concatenate(np.array(val_list)).reshape(-1)

    # For now we match to the test set
    df_seq = pd.DataFrame({'power_pred': val_list}, index = dt_list)

    # For now we join the df_seq on the dt_test to pad the missing test days | we are missing values for some days (when sequence is not full length)
    df_index = pd.DataFrame(index = dt_test)
    df_merged = df_index.join(df_seq)

    # We get the predictions in the correct shape
    Y_pred = np.concatenate(df_merged.values)

    return(Y_pred)

def predict_step_ahead_LSTM(model, X_test, timestep_idx, col_idx):
    # We grab a sample for a given dt_idx | (1, features, n_lag,)
    X_sample = X_test[timestep_idx, :, :].reshape((1, X_test.shape[1], X_test.shape[2]))

    # We make a prediction
    Y_pred_recursive = model.predict(X_sample)
    
    # We set the predicted power equal to the lagged power in the next step | This makes the forecast recursive
    # Check here that we don't overwrite the starting power value for the next test day
    # For every new test day we update the initial power so that we don't have to worry about overwriting
    
    # We have to have a boundary condition for the last day (value) in the test set
    if timestep_idx + 1 >= X_test.shape[0]:
        pass
    # We also have a boundary such that every 48th timestep we don't update (indicating the start of a new day)
    elif (timestep_idx + 1) % 48 == 0:
        pass
    else:
        # We set the whole power sequence in the next step equal to the shifted one we have (we set the last value to NaN)
        X_test[timestep_idx + 1, :, col_idx] = shift1(X_test[timestep_idx, :, col_idx], -1)

        # We then update the last value (NaN) to our new prediction
        X_test[timestep_idx + 1, -1, col_idx] = Y_pred_recursive

        # This way we make sure to carry over the correct power sequences for t-1, t-2, t-3 etc.

    return(Y_pred_recursive, X_test)

# NB. this function is not used in the final results.
def recursive_predict_test_days_LSTM(model, X_test, dt_test, col_idx):
    # We get forecast days
    forecast_days = dt_test[dt_test.time == datetime.time(0, 0)]

    # We convert X_test to a dataframe to use datetimes for getting the datetimes for each day
    df_index = pd.DataFrame(index = dt_test)

    # Based on the way we build the LSTM features the first feature: 0 will always be observations e.g. power
    # The column containing power_t-1 will then be the [n_lag]th column e.g for us the 144th column which is simply idx [-1] in the first feature

    # We create the recursive forecast for all forecasts days
    dt_list = []
    val_list = []
    print("Predicting recursively for test days...")
    for i, day in enumerate(forecast_days):
        #print("Day", i+1, "out of", len(forecast_days))

        # We get a dynamic dt_range (allows for test days with nans) which allows test days to not have full length
        dt_range = df_index.loc[day.strftime('%Y-%m-%d')].index
        dt_list.append(dt_range)

        # We grab the corresponding indices in X_test [samples, time, features]
        both = set(dt_test).intersection(set(dt_range))
        dt_range_idx = [list(dt_test).index(x) for x in both]
        dt_range_idx.sort()

        # Before we make recursive predictions for a new day we update the whole power sequence | it would be overwritten from predict_step_ahead_LSTM for the previous test day otherwise
        # X_test[dt_range_idx[0], :, 0] = Y_test[dt_range_idx[0], :]

        # We create the recursive forecast using the predict_step_ahead_LSTM function
        predictions = []
        for timestep_idx in dt_range_idx:
            #print("Power at dt before:", X_test[timestep_idx, -1, 0])
            
            # We make a power prediction and update recursively for a given datetime
            Y_pred_recursive, X_test = predict_step_ahead_LSTM(model, X_test, timestep_idx, col_idx)

            #print("Prediction:", Y_pred_recursive)
            #if timestep_idx + 1 < X_test.shape[0]:
                #print("Power at dt+1 after:", X_test[timestep_idx+1, -1, 0])

            # We save the prediction
            predictions.append(Y_pred_recursive)

        val_list.append(predictions)
    
    dt_list = np.concatenate(np.array(dt_list)).reshape(-1)
    val_list = np.concatenate(np.array(val_list)).reshape(-1)

    # For now we match to the test set
    df_LSTM = pd.DataFrame({'power_pred': val_list}, index = dt_list)

    # For now we join the df_LSTM on the dt_test to pad the missing test days | we are missing values for some days (when sequence is not full length)
    df_index = pd.DataFrame(index = dt_test)
    df_merged = df_index.join(df_LSTM)

    # We get the predictions in the correct shape
    Y_pred = np.concatenate(df_merged.values)

    return(Y_pred)