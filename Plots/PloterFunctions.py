import matplotlib.pyplot as plt
from Plots import InitParams
import pandas as pd
from Networks.RNN import RNN, create_training_dataset_with_squence
import numpy as np


def plot_net_training_process():
    # Read log file
    dataDf = pd.read_csv("training.log")
    # Separate values
    epochs = dataDf["epoch"].values
    val_loss = dataDf["val_loss"].values
    loss = dataDf["loss"].values
    # Plot
    plt.plot(epochs, loss, color="darkblue")
    plt.plot(epochs, val_loss, color="crimson")
    plt.legend(("loss value", "validation loss value"), ncol=1, loc="best")
    # Save plot
    plt.savefig("trainprocess.pdf")


def plot_future_prediction(trainData, predData, scaler, outputIdxColumn, lookback):
    # FOR FUTURE REFACTORING
    # NEED REFACTORING SO MUCH. JUST FOR FAST PLOT PURPOSE
    seqTrainDataX, seqTrainDataY = create_training_dataset_with_squence(trainData, outputIdxColumn, lookback)
    seqFutureDataX, seqFutureDataY = create_training_dataset_with_squence(predData, outputIdxColumn, lookback)
    rnn = RNN(0)
    rnn.load_network()
    predTrain = rnn.predict(seqTrainDataX)
    predFuture = rnn.predict(seqFutureDataX)


    plt.plot(np.arange(len(seqTrainDataX)), seqTrainDataY, color='darkblue')
    plt.plot(np.arange(len(seqTrainDataX), len(seqTrainDataX)+len(seqFutureDataX)), seqFutureDataY, color="crimson")
    plt.plot(np.arange(len(seqTrainDataX)), predTrain)
    plt.plot(np.arange(len(seqTrainDataX), len(seqTrainDataX)+len(seqFutureDataX)), predFuture)
    plt.legend(("real train y", "RDD predicted train y", "real prediction y", "RNN predicted y"), ncol=1, loc="best")

    plt.savefig("prediction.pdf")



