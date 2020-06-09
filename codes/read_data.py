import numpy as np
import pandas as pd


def read_from_csv(data_set_id):
    """
    read data from the CSV files into numpy
    @param data_set_id: [1]:llvm_input [2]:noc_CM_log [3]:sort_256
    @return: numpy array X_full, X, Y
    """
    if data_set_id == 'llvm':
        df = pd.read_csv('../train_data/llvm_input.csv', sep=',', header=None)
        data = df.values
        X_full = [] # contains both the 11 x and 2 y
        X = [] # contains only 11 x
        Y = [] # contains only 2 y
        for i in data:
            str_list = i[0].split(";")
            sample = [eval(j) for j in str_list]
            sample_x = sample[:-2]
            sample_y = sample[-2:]
            X_full.append(sample)
            X.append(sample_x)
            Y.append(sample_y)
        X_full = np.array(X_full, dtype=np.float64)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        # range0 = max(Y[:, 0]) - min(Y[:, 0])
        # range1 = max(Y[:, 1]) - min(Y[:, 1])
        # for i in range(len(Y)):
        #     Y[i][0] = Y[i][0]
        #     Y[i][1] = Y[i][1]
        return X_full, X, Y
    elif data_set_id == 'noc':
        df = pd.read_csv('../train_data/noc_CM_log.csv', sep=',', header=None)
        data = df.values
        X_full = [] # contains both the 11 x and 2 y
        X = [] # contains only 11 x
        Y = [] # contains only 2 y
        for i in data[1:]:
            str_list = i[0].split(";")
            sample = [eval(j) for j in str_list]
            sample_x = sample[:-2]
            sample_y = sample[-2:]
            X_full.append(sample)
            X.append(sample_x)
            Y.append(sample_y)
        X_full = np.array(X_full, dtype=np.float64)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        range0 = (max(Y[:, 0]) - min(Y[:, 0])) / 50
        range1 = (max(Y[:, 1]) - min(Y[:, 1])) / 50
        # make the data looks the same as the data in the paper
        for i in range(len(Y)):
            Y[i][0] = Y[i][0] * (-1) / range0
            Y[i][1] = Y[i][1] / range1
        return X_full, X, Y
    elif data_set_id == 'snw':
        df = pd.read_csv('../train_data/sort_256.csv', sep=',', header=None)
        data = df.values
        X_full = [] # contains both the 11 x and 2 y
        X = [] # contains only 11 x
        Y = [] # contains only 2 y
        for i in data:
            str_list = i[0].split(";")
            sample = [eval(j) for j in str_list]
            sample_x = sample[:-2]
            sample_y = sample[-2:]
            X_full.append(sample)
            X.append(sample_x)
            Y.append(sample_y)
        X_full = np.array(X_full, dtype=np.float64)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        # print(Y)
        # print(max(Y[:, 0]), min(Y[:, 0]))
        # print(max(Y[:, 0]) - min(Y[:, 0]))
        # print(max(Y[:, 1]), min(Y[:, 1]))
        # print(max(Y[:, 1]) - min(Y[:, 1]))
        range0 = (max(Y[:, 0]) - min(Y[:, 0])) / 100
        range1 = (max(Y[:, 1]) - min(Y[:, 1])) / 100

        # make the data looks the same as the data in the paper
        for i in range(len(Y)):
            Y[i][0] = Y[i][0] * (-1) / range0
            Y[i][1] = Y[i][1] / range1
        return X_full, X, Y
    else:
        raise Exception('No dataset named ' + str(data_set_id))

