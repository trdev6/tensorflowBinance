import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import ObtenerDatos
import random
from conectbd import create_connection
from conectbd import InsertDataTensorflow
from tensorflow import keras
from precision import Precision

# set seed, so we can get the same results after rerunning several times
from ObtenerDatos import data

np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def create_model(sequence_length, units=ObtenerDatos.UNITS, cell=LSTM, n_layers=ObtenerDatos.N_LAYERS,
                 dropout=ObtenerDatos.DROPOUT,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


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


def ReporteTensorflow(symbol, update_model, last_update, epoch, params, precision, mae, priceMin,
                      priceMax, priceClose):
    # symbol = "BTCUSD"
    # update_model = "18/11/2020 15:50:40"
    # last_update = "18/11/2020 15:50:40"
    # params = "tiker=ETH adad = adad "
    # mean_absolute_error = "0.0001"
    # priceMin = "120.001"
    # priceMax = "150.0003"
    # priceClose = " 140.004"

    task = (symbol, update_model, last_update, epoch, params, precision, mae, priceMin,
            priceMax, priceClose)
    InsertDataTensorflow(create_connection(), task)


dropuot_n = str(ObtenerDatos.DROPOUT).replace(".", "_")
test_size_n = str(ObtenerDatos.TEST_SIZE).replace(".", "_")

# model name to save, making it as unique as possible based on parameters
model_name = f"{ObtenerDatos.date_now}_{ObtenerDatos.ticker_n}-{ObtenerDatos.LOSS}-{ObtenerDatos.OPTIMIZER}" \
             f"-{ObtenerDatos.CELL.__name__}-seq-{ObtenerDatos.N_STEPS}-step-{ObtenerDatos.LOOKUP_STEP}" \
             f"-layers-{ObtenerDatos.N_LAYERS}-units-{ObtenerDatos.UNITS}-dropout-{dropuot_n}" \
             f"-batchsize-{ObtenerDatos.BATCH_SIZE}-test_size-{test_size_n}"
if ObtenerDatos.BIDIRECTIONAL:
    model_name += "-b"

    # construct the model
model = create_model(ObtenerDatos.N_STEPS, loss=ObtenerDatos.LOSS, units=ObtenerDatos.UNITS, cell=ObtenerDatos.CELL,
                     n_layers=ObtenerDatos.N_LAYERS, dropout=ObtenerDatos.DROPOUT, optimizer=ObtenerDatos.OPTIMIZER,
                     bidirectional=ObtenerDatos.BIDIRECTIONAL)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
                               save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=ObtenerDatos.BATCH_SIZE,
                    epochs=ObtenerDatos.EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
model.save(os.path.join("results", model_name) + ".h5")

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
precision = Precision()
# Predecir
path_model = "results/" + model_name + ".h5"

# predicted_price = predicted_price_open, predicted_price_high, predicted_price_low, predicted_price_close
new_model = keras.models.load_model(path_model)

# priceOpen = predict(new_model, data)[0]
priceHigh = predict(new_model, data)[1]
priceLow = predict(new_model, data)[2]
priceClose = predict(new_model, data)[3]

# Registrar datos del modelo
ReporteTensorflow(symbol=ObtenerDatos.ticker_n, update_model=f'{ObtenerDatos.from_date} - {ObtenerDatos.to_date}',
                  last_update=ObtenerDatos.fecha_hora,
                  epoch=ObtenerDatos.EPOCHS,
                  params=
                  f'LOSS={ObtenerDatos.LOSS} '
                  f'OPTIMIZER={ObtenerDatos.OPTIMIZER} '
                  f'CELL={ObtenerDatos.CELL.__name__} '
                  f'SEQ={ObtenerDatos.N_STEPS} '
                  f'STEP={ObtenerDatos.LOOKUP_STEP} '
                  f'LAYERS={ObtenerDatos.N_LAYERS} '
                  f'UNITS={ObtenerDatos.UNITS} '
                  f'DROPOUT={ObtenerDatos.DROPOUT} '
                  f'batchsize={ObtenerDatos.BATCH_SIZE} '
                  f'test_size={ObtenerDatos.TEST_SIZE} ',
                  precision=precision,
                  mae=float("%.4f" % mean_absolute_error),
                  priceMin=float("%.2f" % priceLow),
                  priceMax=float("%.2f" % priceHigh),
                  priceClose=float("%.2f" % priceClose))
