import pandas as pd
import os
from sklearn.externals import joblib


def load_data(fileNames):
    dfs = []
    for name in fileNames:
        try:
            dfs.append(pd.read_csv(os.path.join("Res", name), engine='python'))
        except FileNotFoundError as e:
            print("An exception occurred while trying to load the data file")
            print("Exception message:")
            print(e.args)
    return dfs


def save_scalers(scalerX, scalerY):
    scalerX_filename = "scalerX.save"
    scalerY_filename = "scalerY.save"
    joblib.dump(scalerX, scalerX_filename)
    joblib.dump(scalerY, scalerY_filename)


def load_scalers():
    scalerX_filename = "scalerX.save"
    scalerY_filename = "scalerY.save"
    scalerX= joblib.load(scalerX_filename)
    scalerY = joblib.load(scalerY_filename)
    return scalerX, scalerY
