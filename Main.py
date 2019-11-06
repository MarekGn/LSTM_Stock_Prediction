from Networks.NetworkTools import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from Networks.RNN import RNN, create_training_dataset_with_squence
from sklearn.preprocessing import MinMaxScaler
from Plots.PloterFunctions import plot_net_training_process, plot_future_prediction
import matplotlib.pyplot as plt
import numpy as np


def train_LSTM_network(lookback, outputColIdx, batchSize, trainDataPart, epoch, trainData):
    # Make data sequences proper for LSTM network
    seqDataX, seqDataY = create_training_dataset_with_squence(trainData, outputColIdx, lookback)

    # Spliting data to train data and validation data
    X_train, X_test, y_train, y_test = train_test_split(seqDataX, seqDataY, test_size=1-trainDataPart, shuffle=True)
    print("TrainX {}    TrainY {}   TestX {}    TestY {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))

    # Create the RNN network
    rnn = RNN(lookback)

    # Configuring network (Build layer, add optimizers)
    rnn.create_model(X_train.shape[2])
    rnn.configure()

    # Network training
    rnn.train(batchSize=batchSize, epochs=epoch, x=X_train, y=y_train, valX=X_test, valY=y_test)


if __name__ == '__main__':
    ######     DATA    ######
    # Names of files that we want to use
    fileNames = ["dji_filtered.csv", "gold_filtered.csv", "mcsf_filtered.csv", "oil_filtered.csv",
                 "shanghai_filtered.csv",
                 "sp500_filtered.csv"]
    # Number of days that rnn network lookback while training
    lookback = 30
    # Name of output dataframe column
    outputColName = "MCSFClose"
    # Number of training records checked before network weights update
    batchSize = 32
    # Percent of training data from all data records (Rest is for testing)
    trainDataPart = 0.9
    # Number of epochs
    epoch = 100
    # Part of data for prediction in days (not for training)
    pred_part = 300

    ######     CALCULATIONS    ######
    # Loading files as array of dataframes
    dataDfArr = load_data(fileNames)

    # Merging all dataframes to one dataframe
    dataDf = pd.concat(dataDfArr, axis=1)

    # Splitting dataframes for training and for future prediction
    trainDf = dataDf[0:-pred_part]
    predDf = dataDf[-pred_part-lookback:]

    # Create data scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the data
    scaledTrainData = scaler.fit_transform(trainDf.loc[:, ].values)
    scaledPredData = scaler.transform(predDf.loc[:, ].values)

    plot_future_prediction(scaledTrainData, scaledPredData, scaler, dataDf.columns.get_loc(outputColName), lookback)
    # train_LSTM_network(lookback, dataDf.columns.get_loc(outputColName), batchSize, trainDataPart, epoch, scaledTrainData)
    # plot_net_training_process()





