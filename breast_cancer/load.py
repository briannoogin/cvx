import numpy as np
import os
import pandas as pd

def load_data():
    f = pd.read_csv('wdbc.data', delim_whitespace=True)
    print(f.loc[0])
    df = f.values.tolist()

    x_train, y_train, x_test, y_test = [], [], [], []
    num_data = len(f)
    print('Num rows: ', num_data)

    for x in range(num_data):
        row = df[x][0].split(',')

        # append radius of tumor as regression task
        y_train.append(row[2])

        # all pieces of data are used besides str name, and origin (last 2 indices)
        row_of_data = [row[i] for i in range(3,11)]

        # deal with only malignant tumors, so we don't fit bad data
        if row[1] == 'B':
            y_train = y_train[:-1]
            continue
        else:
            # convert to float
            row_of_data = [float(num) for num in row_of_data]

        x_train.append(row_of_data)

    # now we do 95% train / test split
    size = len(x_train)
    split = int(size / 20)
    
    x_test = x_train[:split]
    x_train = x_train[split:]
    
    y_test = y_train[:split]
    y_train = y_train[split:]

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)

