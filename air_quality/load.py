import numpy as np
import os
import pandas as pd

def load_data(path):
    f = pd.read_table(path, header=None, delim_whitespace=True)
    train, test = [], []
    
    num_data = len(f)
    for x in range(0, num_data):
        test.append(f.loc[x][0])

        # all pieces of data are used besides str name, and origin (last 2 indices)
        row_of_data = [f.loc[x][i] for i in range(1,15)]

        train.append(row_of_data)

    return train, test
