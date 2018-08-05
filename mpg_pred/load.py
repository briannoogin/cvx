import numpy as np
import os
import pandas as pd

def load_data():
    f = pd.read_table('mgp.txt', delim_whitespace=True)
    print(f.loc[0])

    x_train, y_train, x_test, y_test = [], [], [], []
    num_data = len(f)
    for x in range(num_data):
        y_train.append(f.loc[x][0])

        # all pieces of data are used besides str name, and origin (last 2 indices)
        row_of_data = [f.loc[x][i] for i in range(1,5)]
        if row_of_data[2] == '?':
            y_train = y_train[:-1]
            continue
        else:
            row_of_data = [float(num) for num in row_of_data]

        x_train.append(row_of_data)

    # now we do 95% train / test split
    size = len(x_train)
    split = int(size / 15)
    
    x_test = x_train[:split]
    x_train = x_train[split:]
    
    y_test = y_train[:split]
    y_train = y_train[split:]

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)
