import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
import ObtenerDatos
from ObtenerDatos import data
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional


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


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-ObtenerDatos.LOOKUP_STEP],
                      y_pred[ObtenerDatos.LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-ObtenerDatos.LOOKUP_STEP],
                      y_test[ObtenerDatos.LOOKUP_STEP:]))

    precision = float(accuracy_score(y_test, y_pred)) * 100
    return f'{"%.2f" % precision}%'


# new_model = keras.models.#load_model(path_model)

# plot_graph(new_model, data)
def Precision():
    dropuot_n = str(ObtenerDatos.DROPOUT).replace(".", "_")
    test_size_n = str(ObtenerDatos.TEST_SIZE).replace(".", "_")
    model_name = f"{ObtenerDatos.date_now}_{ObtenerDatos.ticker_n}-{ObtenerDatos.LOSS}-{ObtenerDatos.OPTIMIZER}" \
                 f"-{ObtenerDatos.CELL.__name__}-seq-{ObtenerDatos.N_STEPS}-step-{ObtenerDatos.LOOKUP_STEP}" \
                 f"-layers-{ObtenerDatos.N_LAYERS}-units-{ObtenerDatos.UNITS}-dropout-{dropuot_n}" \
                 f"-batchsize-{ObtenerDatos.BATCH_SIZE}-test_size-{test_size_n}"
    if ObtenerDatos.BIDIRECTIONAL:
        model_name += "-b"
    path_model = "results/" + model_name + ".h5"

    # construct the model

    model = create_model(ObtenerDatos.N_STEPS, loss=ObtenerDatos.LOSS, units=ObtenerDatos.UNITS, cell=ObtenerDatos.CELL,
                         n_layers=ObtenerDatos.N_LAYERS, dropout=ObtenerDatos.DROPOUT, optimizer=ObtenerDatos.OPTIMIZER,
                         bidirectional=ObtenerDatos.BIDIRECTIONAL)

    precision = get_accuracy(model, data)

    return precision
