from Networks.NetworkTools import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from Networks.RNN import RNN

if __name__ == '__main__':
    ## DATA ##
    # Names of files that we want to use
    fileNames = ["dji_filtered.csv", "gold_filtered.csv", "mcsf_filtered.csv", "oil_filtered.csv",
                 "shanghai_filtered.csv",
                 "sp500_filtered.csv"]
    # Number of days that rnn network lookback while training
    lookback = 30
    # Name of output dataframe column
    outputColName = "MCSFClose"
    # Number of training records checked before network weights update
    batchSize = 200
    # Percent of training data from all data records (Rest is for testing)
    trainDataPart = 0.8


    ## CALCULATIONS ##
    # Loading files as array of dataframes
    dataDf = load_data(fileNames)
    # Merging all dataframes to one dataframe
    dataDf = pd.concat(dataDf, axis=1)
    # Spliting data to train data and validation data
    trainDf, testDf = train_test_split(dataDf, train_size=trainDataPart, test_size=1-trainDataPart, shuffle=False)
    print("Train and Test data size", len(trainDf), len(testDf))
    # Create the RNN network
    rnn = RNN(lookback)
    # Scale the data
    trainData = trainDf.loc[:, ].values
    testData = testDf.loc[:, ].values
    trainData = rnn.scale_and_fit_data(trainData)
    testData = rnn.scale_data(testData)
    # Creating sequence data for rnn network
    rnn.create_training_dataset_with_squence(trainData, dataDf.columns.get_loc(outputColName))
    # Configuring network (Build layer, add optimizers)
    rnn.create_model()
    rnn.configure()
    # Network training
    rnn.train(batchSize=batchSize, epochs=10)
