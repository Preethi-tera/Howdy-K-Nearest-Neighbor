import multiprocessing as mp
import ctypes
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import random
from itertools import combinations
import matplotlib.pyplot as plt

from test_model import test_model

# Comma-separated values
df = pd.read_csv("cleveland.csv")

# Rename 'num' column to 'disease' and change 1,2,3,4 to 1
df = df.rename({"num": "disease"}, axis=1)
df["disease"] = df.disease.apply(lambda x: min(x, 1))

# Fix some of the question marks not being interpreted as null
df = df[df["ca"] != "?"]
df = df[df["thal"] != "?"]

df["ca"] = df["ca"].astype("float")
df["thal"] = df["thal"].astype("float")

std_df = df.copy()

for column in df.columns[:-1]:
    std_df[column] = (df[column] - df[column].mean()) / df[column].std()

SAMPLE_SIZE = 30


# K needs to be lower than our sample dataset, which stays small
MAX_K = 20
ATTRIBUTE_NUMBER = 6

# the origingal dataframe is df, store the columns/dtypes pairs
df_dtypes_dict = dict(list(zip(std_df.columns, std_df.dtypes)))

# declare a shared Array with data from df
mparr = mp.Array(ctypes.c_double, std_df.values.reshape(-1))

# create a new df based on the shared array
df_shared = pd.DataFrame(
    np.frombuffer(mparr.get_obj()).reshape(df.shape), columns=df.columns
).astype(df_dtypes_dict)


def test_adapter(columns: tuple):
    res = {k: test_model(df_shared, list(columns), k) for k in range(1, 20 + 1)}
    key = max(res, key=res.get)
    print(f"Max F: {key:2} - {res[key]:6.5}\t {columns}")
    return {columns: (key, res[key])}


if __name__ == "__main__":
    with mp.Pool(15) as p:
        print("starting thread pool...")
        test_columns = []
        for i in range(1, 10 + 1):
            combos = list(combinations(std_df.columns[:-1], i))
            random.shuffle(combos)
            test_columns += combos[:10]

        res = p.map_async(test_adapter, test_columns)
        p.close()
        p.join()

        for answer in res.get():
            for column, values in answer.items():
                k_val, mean_f = values
                if mean_f > 0.65:
                    print(f"{column}, {k_val}, {mean_f}")
