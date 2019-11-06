import numpy as np
from Interface.NetworkInterface import Network
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Activation, Dropout
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import optimizers, backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

class RNN(Network):
    def __init__(self, lookback):
        self.lookback = lookback
        self.model = None
        get_custom_objects().update({'bent': Bent(bent)})

    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction

    def create_model(self, inputShape):
        self.model = Sequential()
        self.model.add(LSTM(units=128, input_shape=(self.lookback, inputShape), return_sequences=True, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=64, return_sequences=False, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=64, activation='bent'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(units=32, activation='bent'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(units=1, activation='bent'))

    def configure(self):
        optimizer = optimizers.RMSprop(lr=1e-4)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def train(self, x, y, valX, valY, batchSize=32, epochs=100):
        # Callbacks
        es = EarlyStopping(monitor='loss', patience=15000, min_delta=1e-8)
        csv_logger = CSVLogger('training.log')
        valChp = ModelCheckpoint(filepath="dnnvalidation.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        trainChp = ModelCheckpoint(filepath="dnntrain.h5", monitor='loss', verbose=0, save_best_only=True, mode='min')
        # Training
        self.model.fit(x, y, validation_data=(valX, valY), epochs=epochs, batch_size=batchSize, verbose=2, shuffle=True, callbacks=[es, csv_logger ,valChp, trainChp])

    def load_network(self, suffix="train"):
        self.model = load_model("dnn{}.h5".format(suffix))


def create_training_dataset_with_squence(trainData, outputIdx, lookback):
    # outputIdx is the index of out output column
    dim_0 = trainData.shape[0] - lookback
    dim_1 = trainData.shape[1]
    x = np.zeros((dim_0, lookback, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = trainData[i:lookback + i]
        y[i] = trainData[lookback + i, outputIdx]
    print("Shape of Input and Output data ", np.shape(x), np.shape(y))
    return x, y


class Bent(Activation):

    def __init__(self, activation, **kwargs):
        super(Bent, self).__init__(activation, **kwargs)
        self.__name__ = 'bent'


def bent(x):
    return ((K.sqrt(K.pow(x, 2) + 1) - 1) / 2) + x
