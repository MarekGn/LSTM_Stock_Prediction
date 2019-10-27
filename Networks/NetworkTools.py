import pandas as pd
import os


def load_data(fileNames):
    dfs = []
    for name in fileNames:
        try:
            dfs.append(pd.read_csv(os.path.join("Res", name), engine='python'))
        except FileNotFoundError as e:
            print("An exception occured while trying to load the data file")
            print("Exception message:")
            print(e.args)
    return dfs