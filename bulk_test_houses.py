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
df = pd.read_csv("loan_sanction_train.csv")

#  Change the loan status to a binary value
df["Loan_Status"] = df.Loan_Status.apply(lambda x: 1 if x == "Y" else 0).astype("int")
df["Dependents"] = df["Dependents"].apply(lambda x: 3 if x == "3+" else x)
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)
df["Married"] = df["Married"].apply(lambda x: 1 if x == "Yes" else 0)
df["Self_Employed"] = df["Self_Employed"].apply(lambda x: 1 if x == "Yes" else 0)
df["Education"] = df["Education"].apply(lambda x: 1 if x == "Graduate" else 0)
property_map = {"Urban": 1, "Rural": 2, "Semiurban": 3}
df["Property_Area"] = df["Property_Area"].apply(lambda x: property_map[x])
std_df = df.copy()

# Standardize all but the disease column at the end
for column in [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]:
    std_df[column] = (df[column] - df[column].mean()) / df[column].std()

# Larger sample size for larger training set
SAMPLE_SIZE = 60

# K needs to be lower than our sample dataset, which stays small
MAX_K = 40
ATTRIBUTE_NUMBER = 6

# create a new df based on the shared array
df_shared = std_df


def test_adapter(columns: tuple):
    res = {
        k: test_model(
            df_shared, list(columns), k, SAMPLE_SIZE, answer_column="Loan_Status"
        )
        for k in range(1, 20 + 1)
    }
    key = max(res, key=res.get)
    print(f"Max F: {key:2} - {res[key]:6.5}\t {columns}")
    return {columns: (key, res[key])}


if __name__ == "__main__":
    with mp.Pool(15) as p:
        print("starting thread pool...")
        test_columns = []
        for i in [1, 2]:
            combos = list(combinations(std_df.columns[1:-1], i))
            random.shuffle(combos)
            test_columns += combos

        res = p.map_async(test_adapter, test_columns)
        p.close()
        p.join()

        for answer in res.get():
            for column, values in answer.items():
                k_val, mean_f = values
                if mean_f > 0.65:
                    print(f"{column}, {k_val}, {mean_f}")
