import matplotlib.pyplot as plt
from Plots import InitParams
import pandas as pd
import numpy as np


def plot_net_training_process(y_bottom=None, y_top=None):
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
    plt.ylim(y_bottom, y_top)
    # Save plot
    plt.savefig("trainprocess.pdf")
    plt.cla()


#######################################################################################################################
    # UNDERMENTIONED FUNCTIONS
    # FOR FUTURE REFACTORING
    # NEED REFACTORING SO MUCH. JUST FOR FAST PLOT PURPOSE
#######################################################################################################################

def plot_train_and_future_prediction(seqTrainX, seqTrainY, seqFutureX, seqFutureY, rnn, scalerY):
    predTrain = rnn.predict(seqTrainX)
    predFuture = rnn.predict(seqFutureX)

    predTrain = scalerY.inverse_transform(predTrain.reshape(-1, 1))
    predFuture = scalerY.inverse_transform(predFuture.reshape(-1, 1))
    seqTrainY = scalerY.inverse_transform(seqTrainY.reshape(-1, 1))
    seqFutureY = scalerY.inverse_transform(seqFutureY.reshape(-1, 1))

    plt.plot(np.arange(len(seqTrainX)), seqTrainY, color='darkblue')
    plt.plot(np.arange(len(seqTrainX)), predTrain, color='lightblue')
    plt.plot(np.arange(len(seqTrainX), len(seqTrainX)+len(seqFutureX)), seqFutureY, color="darkred")
    plt.plot(np.arange(len(seqTrainX), len(seqTrainX)+len(seqFutureX)), predFuture, color='lightcoral')
    plt.legend(("real train y", "RNN predicted train y", "real future y", "RNN future y"), ncol=1, loc="best")

    plt.savefig("train_future_prediction.pdf")
    plt.cla()


def plot_train(seqTrainX, seqTrainY, rnn, scalerY):
    predTrain = rnn.predict(seqTrainX)

    predTrain = scalerY.inverse_transform(predTrain.reshape(-1, 1))
    seqTrainY = scalerY.inverse_transform(seqTrainY.reshape(-1, 1))

    plt.plot(np.arange(len(seqTrainX)), seqTrainY, color='darkblue')
    plt.plot(np.arange(len(seqTrainX)), predTrain, color='lightblue')
    plt.legend(("real train y", "RNN predicted train y"), ncol=1, loc="best")

    plt.savefig("train_prediction.pdf")
    plt.cla()


def plot_future(seqFutureX, seqFutureY, rnn, scalerY):
    predFuture = rnn.predict(seqFutureX)

    predFuture = scalerY.inverse_transform(predFuture.reshape(-1, 1))
    seqFutureY = scalerY.inverse_transform(seqFutureY.reshape(-1, 1))

    plt.plot(np.arange(len(seqFutureX)), seqFutureY, color="darkred")
    plt.plot(np.arange(len(seqFutureX)), predFuture, color='lightcoral')
    plt.legend(("real train y", "RNN predicted train y", "real future y", "RNN future y"), ncol=1, loc="best")

    plt.savefig("future_prediction.pdf")
    plt.cla()




