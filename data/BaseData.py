import pandas as pd
import numpy as np

def getData(tiker="GOOGL", window=10, des3=False):
    tt = "Low"
    tt1 = "Low"
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
    df1 = pd.read_csv('./data/datasets/' + str(tiker) + '1.csv')
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
