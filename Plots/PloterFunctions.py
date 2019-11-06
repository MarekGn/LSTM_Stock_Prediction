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


def plot_future_prediction(seqTrainX, seqTrainY, seqFutureX, seqFutureY, rnn, scalerY):
    # FOR FUTURE REFACTORING
    # NEED REFACTORING SO MUCH. JUST FOR FAST PLOT PURPOSE
    predTrain = rnn.predict(seqTrainX)
    predFuture = rnn.predict(seqFutureX)

    predTrain = scalerY.inverse_transform(predTrain.reshape(-1, 1))
    predFuture = scalerY.inverse_transform(predFuture.reshape(-1, 1))
    seqTrainY = scalerY.inverse_transform(seqTrainY.reshape(-1, 1))
    seqFutureY = scalerY.inverse_transform(seqFutureY.reshape(-1, 1))


    plt.plot(np.arange(len(seqTrainX)), seqTrainY, color='darkblue')
    plt.plot(np.arange(len(seqTrainX), len(seqTrainX)+len(seqFutureX)), seqFutureY, color="crimson")
    plt.plot(np.arange(len(seqTrainX)), predTrain)
    plt.plot(np.arange(len(seqTrainX), len(seqTrainX)+len(seqFutureX)), predFuture)
    plt.legend(("real train y", "RDD predicted train y", "real prediction y", "RNN predicted y"), ncol=1, loc="best")

    plt.savefig("prediction.pdf")



