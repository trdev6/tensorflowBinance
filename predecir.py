import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential

import ObtenerDatos
import conectbd
from ObtenerDatos import N_STEPS, data
from tensorflow import keras


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-ObtenerDatos.N_STEPS:]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price_open = column_scaler["open"].inverse_transform(prediction)[0][0]
    predicted_price_high = column_scaler["high"].inverse_transform(prediction)[0][0]
    predicted_price_low = column_scaler["low"].inverse_transform(prediction)[0][0]
    predicted_price_close = column_scaler["close"].inverse_transform(prediction)[0][0]
    predicted_price = [predicted_price_open, predicted_price_high, predicted_price_low, predicted_price_close]
    return predicted_price


def ReporteTensorflow(symbol, update_model, last_update, params, mean_absolute_error, priceMin,
                      priceMax, priceClose):
    # symbol = "BTCUSD"
    # update_model = "18/11/2020 15:50:40"
    # last_update = "18/11/2020 15:50:40"
    # params = "tiker=ETH adad = adad "
    # mean_absolute_error = "0.0001"
    # priceMin = "120.001"
    # priceMax = "150.0003"
    # priceClose = " 140.004"

    task = (symbol, update_model, last_update, params, mean_absolute_error, priceMin,
            priceMax, priceClose)
    conectbd.InsertDataTensorflow(conectbd.create_connection(), task)


dropuot_n = str(ObtenerDatos.DROPOUT).replace(".", "_")
test_size_n = str(ObtenerDatos.TEST_SIZE).replace(".", "_")
model_name = f"{ObtenerDatos.date_now}_{ObtenerDatos.ticker_n}-{ObtenerDatos.LOSS}-{ObtenerDatos.OPTIMIZER}" \
             f"-{ObtenerDatos.CELL.__name__}-seq-{ObtenerDatos.N_STEPS}-step-{ObtenerDatos.LOOKUP_STEP}" \
             f"-layers-{ObtenerDatos.N_LAYERS}-units-{ObtenerDatos.UNITS}-dropout-{dropuot_n}" \
             f"-batchsize-{ObtenerDatos.BATCH_SIZE}-test_size-{test_size_n}"
if ObtenerDatos.BIDIRECTIONAL:
    model_name += "-b"
path_model = "results/" + model_name + ".h5"

new_model = keras.models.load_model(path_model)

# priceOpen = predict(new_model, data)[0]
priceHigh = predict(new_model, data)[1]
priceLow = predict(new_model, data)[2]
priceClose = predict(new_model, data)[3]

print(priceClose)