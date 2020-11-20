import os
import time
# from yahoo_fin import stock_info as si
from collections import deque
import numpy as np
# import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM
import investpy_data
from tensorboard import data
import configparser
import datetime

configuracion = configparser.ConfigParser()
# Leer el archivo de configuración:
def ComprobarConfig():
    if os.path.isfile('configuracion.cfg'):
        configuracion.read('configuracion.cfg')
    else:
        print("No se encontro archivo de configuracion, se creara uno nuevo")
        configuracion['General'] = {'from_date':
                                        '01/01/2019',
                                    'to_date': '11/11/2020', 'N_STEPS': '30',
                                    'LOOKUP_STEP': '1', 'TEST_SIZE': '0.2', 'FEATURE_COLUMNS': ["open", "high", "low",
                                                                                                "close", "volume"],
                                    'N_LAYERS': '4', 'CELL': 'LSTM', 'UNITS': '256', 'DROPOUT': '0.4',
                                    'BIDIRECTIONAL': 'False', 'LOSS': 'mae', 'OPTIMIZER': 'adam',
                                    'BATCH_SIZE': '64', 'EPOCHS': '400', 'ticker': 'Ethereum', 'tipoActivo': 'cripto'}
        with open('configuracion.cfg', 'w') as archivoconfig:
            configuracion.write(archivoconfig)
        configuracion.read('configuracion.cfg')

def load_data(from_date, to_date, ticker, n_steps=30, scale=True, shuffle=True, lookup_step=1,
              test_size=0.2, feature_columns=['open', 'high', 'low', 'close', 'volume']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    # if isinstance(ticker, str):
    # load it from yahoo_fin library

    if tipoActivo == "cripto":
        df = investpy_data.dataHistoricalCripto(ticker, from_date, to_date)
    if tipoActivo == "currency":
        df = investpy_data.dataHistoricalCurrencyCross(ticker, from_date, to_date)
    # elif isinstance(ticker, pd.DataFrame):
    # already loaded, use it directly
    #    df = ticker
    # else:
    #    raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {'df': df.copy()}
    # print(result)
    # we will also return the original dataframe itself
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                shuffle=shuffle)
    # return the result
    return result

ComprobarConfig()

general = configuracion['General']
# fecha
from_date = str(general['from_date'])
to_date = str(general['to_date'])


# Window size or the sequence length
N_STEPS = int(general['N_STEPS'])
# Lookup step, 1 is the next day
LOOKUP_STEP = int(general['LOOKUP_STEP'])
# test ratio size, 0.2 is 20%
TEST_SIZE = float(general['TEST_SIZE'])
# features to use
FEATURE_COLUMNS = eval(general['FEATURE_COLUMNS'])
N_LAYERS = int(general['N_LAYERS'])
# 256 LSTM neurons
UNITS = int(general['UNITS'])
# 40% dropout
DROPOUT = float(general['DROPOUT'])
# whether to use bidirectional RNNs
BIDIRECTIONAL = bool(general['BIDIRECTIONAL'])
LOSS = str(general['LOSS'])
OPTIMIZER = str(general['OPTIMIZER'])
BATCH_SIZE = int(general['BATCH_SIZE'])
EPOCHS = int(general['EPOCHS'])

date_now = time.strftime("%Y-%m-%d")
now = datetime.datetime.now()
fecha_hora = f'{now.day}/{now.month}/{now.year}, {now.hour}:{now.minute}:{now.second}'
# LSTM cell
CELL = LSTM
# BIDIRECTIONAL = False
tipoActivo = str(general['tipoActivo'])
ticker = str(general['ticker'])
ticker_n = ticker.replace("/", "-")
ticker_data_filename = os.path.join("data", f"{ticker_n}_{date_now}.csv")


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

# load the data
data = load_data(from_date, to_date, ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)
# save the dataframe
data["df"].to_csv(ticker_data_filename)


