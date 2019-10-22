#Load Packages
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.callbacks import EarlyStopping

class RNN:
    def __init__(self, lookback):
        self.trainX = None
        self.trainY = None
        self.predX = None
        self.accuracy = None
        self.model = None
        self.lookback = lookback
        self.layer_sizes = ''
        self.Yscaler = MinMaxScaler(feature_range=(0, 1))
        self.Xscaler = MinMaxScaler(feature_range=(0, 1))

        get_custom_objects().update({'bent': Bent(bent)})

    def predict(self, x):
        self.create_dataset(x=x, training=False)
        prediction = self.model.predict(self.predX, batch_size=1)
        prediction = self.Yscaler.inverse_transform(prediction)
        return prediction

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=32, input_shape=(1, self.lookback), return_sequences=True, activation='bent'))
        self.model.add(LSTM(units=32, return_sequences=False, activation='bent'))
        self.model.add(Dense(units=32, activation='bent'))
        self.model.add(Dense(units=16, activation='bent'))
        self.model.add(Dense(units=1, activation='bent'))

        self.layer_sizes = ''
        for l in self.model.layers:
            self.layer_sizes += str(l.output_shape)+" "

    def configure(self):
        self.model.compile(optimizer='adam', loss="mean_squared_error")

    def train(self, batch_size=32, epochs=1000):
        es = EarlyStopping(monitor='loss', patience=15000, min_delta=1e-10)
        self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,
                       callbacks=[es])

    def create_training_dataset_with_squence(self, x, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset)-(self.lookback-1)):
            a = x[i:(i+self.lookback), 0]
            dataX.append(a)
            dataY.append(dataset[i + (self.lookback-1), 0])
        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # LSTM network must have input with 3 dimensions
        dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
        return dataX, dataY

    def create_dataset(self, x, y=None, training=False):
        x = x.reshape((-1, 1))
        if y is None:
            y = x.copy()
        y = y.reshape((-1, 1))
        if training is True:
            # Network Training
            y = self.Yscaler.fit_transform(y)
            x = self.Xscaler.fit_transform(x)
            self.trainX, self.trainY = self.create_training_dataset_with_squence(x, y)
        else:
            # Network Predicting
            y = self.Yscaler.transform(y)
            x = self.Xscaler.transform(x)
            self.predX, _ = self.create_training_dataset_with_squence(x, y)

    def check_accuracy(self, x, y):
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x = self.Xscaler.transform(x)
        x, _ = self.create_training_dataset_with_squence(x, y)
        predictedY = self.Yscaler.inverse_transform(self.model.predict(x))
        self.accuracy = np.mean(np.square(y - predictedY))
        print("RNN accuracy is {}".format(self.accuracy))
        return self.accuracy


class Bent(Activation):

    def __init__(self, activation, **kwargs):
        super(Bent, self).__init__(activation, **kwargs)
        self.__name__ = 'bent'


def bent(x):
    return ((K.sqrt(K.pow(x, 2) + 1) - 1) / 2) + x
