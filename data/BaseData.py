import pandas as pd
import numpy as np


def getData(tiker="GOOGL", window=10, des3=False, tiker_test=None):
    if tiker_test is None:
        tiker_test = tiker
    tt = "Close"
    tt1 = "Close"
    # train
    df = pd.read_csv('./data/datasets/' + str(tiker) + '0.csv')
    df = (df[tt].diff() * 100 / df[tt1]).dropna().values.tolist()
    # window = 10
    train_data = []
    for row in range(window, len(df)):
        start = row - window
        end = row
        train_data.append([df[i] for i in range(start, end)])
    train_data = np.array(train_data)
    train_data = train_data.astype('float32')
    if des3:
        train_data = np.reshape(train_data, (len(train_data), window, 1))
    else:
        train_data = np.reshape(train_data, (len(train_data), window))

    # test
    df1 = pd.read_csv('./data/datasets/' + str(tiker_test) + '1.csv')
    df1 = (df1[tt].diff() * 100 / df1[tt1]).dropna().values.tolist()
    test_data = []
    for row in range(window, len(df1)):
        start = row - window
        end = row
        test_data.append([df1[i] for i in range(start, end)])
    test_data = np.array(test_data)
    test_data = test_data.astype('float32')
    if des3:
        test_data = np.reshape(test_data, (len(test_data), window, 1))
    else:
        test_data = np.reshape(test_data, (len(test_data), window))
    return train_data, test_data


def getDataN1(window=10, streams=None, test=False):
    # (data[column1][k]/data[column2][k-shift]) -> (tiker, column1, column2, shift)
    if streams is None:
        streams = []
    train_data = []
    max_shift = 0
    for _, _, _, shft in streams:
        if max_shift < shft:
            max_shift = shft

    length = 0
    for tiker, column1, column2, shift in streams:
        sup_shift = 0
        if shift < max_shift:
            sup_shift = max_shift - shift

        # train
        suff = '0.csv'
        if test:
            suff = '1.csv'
        df = pd.read_csv('./data/datasets/' + str(tiker) + str(suff))
        df = (df[column1].shift(sup_shift) / df[column2].shift(shift + sup_shift) - 1) * 100.0
        df = df.dropna()
        length = df.size - window
        df = df.tolist()


        stream_data = []
        for row in range(window, len(df)):
            start = row - window
            end = row
            # stream_data.append([df[i] for i in range(start, end)])
            for i in range(start, end):
                train_data.append(df[i])
            # train_data.append([df[i] for i in range(start, end)])
        # train_data.append(stream_data)
    train_data = np.array(train_data)
    train_data = train_data.astype('float32')
    train_data = np.reshape(train_data, (len(streams), length, window, 1))
    # train_data = np.swapaxes(train_data, 0, 1)
    train_data1 = np.swapaxes(train_data, 0, 1)
    return train_data1
