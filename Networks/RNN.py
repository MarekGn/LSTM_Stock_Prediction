import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import optimizers, backend as K
from tensorflow.python.keras.callbacks import EarlyStopping

class RNN:
    def __init__(self, lookback):
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.lookback = lookback
        self.model = None
        self.layer_sizes = ''
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        get_custom_objects().update({'bent': Bent(bent)})

    # def predict(self, x):
    #     self.create_dataset(x=x, training=False)
    #     prediction = self.model.predict(self.predX, batch_size=1)
    #     prediction = self.Yscaler.inverse_transform(prediction)
    #     return prediction

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=128, input_shape=(self.lookback, self.trainX.shape[2]), return_sequences=True, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=128, return_sequences=False, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=64, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=32, activation='bent'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation='bent'))

        self.layer_sizes = ''
        for l in self.model.layers:
            self.layer_sizes += str(l.output_shape)+" "

    def configure(self):
        optimizer = optimizers.RMSprop(lr=1e-8, rho=0.9)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def train(self, batchSize=32, epochs=100):
        es = EarlyStopping(monitor='loss', patience=15000, min_delta=1e-6)
        self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=batchSize, verbose=2, shuffle=True, callbacks=[es])

    # def check_mse_accuracy(self, x, y):
    #     x = x.reshape((-1, 1))
    #     y = y.reshape((-1, 1))
    #     x = self.Xscaler.transform(x)
    #     x, _ = self.create_training_dataset_with_squence(x, y)
    #     predictedY = self.Yscaler.inverse_transform(self.model.predict(x))
    #     self.accuracy = np.mean(np.square(y - predictedY))
    #     print("RNN accuracy is {}".format(self.accuracy))
    #     return self.accuracy

    def create_training_dataset_with_squence(self, trainData, outputIdx):
        # outputIdx is the index of out output column
        dim_0 = trainData.shape[0] - self.lookback
        dim_1 = trainData.shape[1]
        x = np.zeros((dim_0, self.lookback, dim_1))
        y = np.zeros((dim_0,))

        for i in range(dim_0):
            x[i] = trainData[i:self.lookback + i]
            y[i] = trainData[self.lookback + i, outputIdx]
        print("shape of Input and Output data ", np.shape(x), np.shape(y))
        self.trainX = x
        self.trainY = y

    def scale_and_fit_data(self, data):
        data = self.scaler.fit_transform(data)
        return data

    def scale_data(self, data):
        data = self.scaler.transform(data)
        return data


class Bent(Activation):

    def __init__(self, activation, **kwargs):
        super(Bent, self).__init__(activation, **kwargs)
        self.__name__ = 'bent'


def bent(x):
    return ((K.sqrt(K.pow(x, 2) + 1) - 1) / 2) + x
