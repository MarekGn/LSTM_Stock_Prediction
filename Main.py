from Networks.NetworkTools import load_data, load_scalers, save_scalers
import pandas as pd
from sklearn.model_selection import train_test_split
from Networks.RNN import RNN, create_training_dataset_with_squence, load_network
from sklearn.preprocessing import MinMaxScaler
from Plots.PloterFunctions import plot_net_training_process, plot_train_and_future_prediction, plot_future, plot_train


def train_LSTM_network(X_train, y_train, X_test, y_test, lookback, batchSize, epoch):
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
                 "sp500_filtered.csv", "events_filtered_scaled.csv"]
    # Number of days that rnn network lookback while training
    lookback = 30
    # Name of output dataframe column
    outputColName = "MCSFClose"
    # Number of training records checked before network weights update
    batchSize = 32
    # Percent of training data from all data records (Rest is for testing)
    trainDataPart = 0.9
    # Number of epochs
    epoch = 520
    # Part of data for prediction in days (not for training)
    pred_part = 180

    ######     CALCULATIONS    ######
    # Loading files as array of dataframes
    dataDfArr = load_data(fileNames)

    # Merging all dataframes to one dataframe
    dataDf = pd.concat(dataDfArr, axis=1)
    # dataDf = dataDf.drop([
    #     "DJIOpen",
    #     "DJIHigh",
    #     "DJILow",
    #     "GoldOpen",
    #     "GoldHigh",
    #     "GoldLow",
    #     "MCSFOpen",
    #     "MCSFHigh",
    #     "MCSFLow",
    #     "OILOpen",
    #     "OILHigh",
    #     "OILLow",
    #     "ShanghaiOpen",
    #     "ShanghaiHigh",
    #     "ShanghaiLow",
    #     "SP500Open",
    #     "SP500High",
    #     "SP500Low"],
    #     axis = 1)
    # Splitting dataframes for training and for future prediction
    trainDf = dataDf[0:-pred_part]
    futureDf = dataDf[-pred_part - lookback:]

    # Create data scaler
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))

    # Scale the data
    scaledTrainData = scalerX.fit_transform(trainDf.loc[:, ].values)
    scaledFutureData = scalerX.transform(futureDf.loc[:, ].values)

    # Fit the scalerY for later use in inverse_transform RNN predicions
    # (predicted Y data have different shape than X data so it need another scaler)
    scalerY.fit(trainDf[outputColName].values.reshape(-1, 1))

    # Save scalers for future use
    save_scalers(scalerX, scalerY)

    # Make data sequences proper for LSTM network
    seqTrainX, seqTrainY = create_training_dataset_with_squence(scaledTrainData, dataDf.columns.get_loc(outputColName), lookback)
    seqFutureX, seqFutureY = create_training_dataset_with_squence(scaledFutureData, dataDf.columns.get_loc(outputColName), lookback)

    # Spliting data to train data and validation data
    X_train, X_test, y_train, y_test = train_test_split(seqTrainX, seqTrainY, test_size=1 - trainDataPart, shuffle=True)
    print("TrainX {}    TrainY {}   TestX {}    TestY {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))


    train_LSTM_network(X_train, y_train, X_test, y_test, lookback, batchSize, epoch)
    # scalerX, scalerY = load_scalers()
    # rnn = load_network(lookback, suffix='trainval')
    # plot_train_and_future_prediction(seqTrainX, seqTrainY, seqFutureX, seqFutureY, rnn, scalerY)
    # plot_future(seqFutureX, seqFutureY, rnn, scalerY)
    # plot_train(seqTrainX, seqTrainY, rnn, scalerY)
    # plot_net_training_process()





