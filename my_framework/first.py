import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_framework.test_model import TestModel, Settings


def get_state(t, trend, window_size):
    window_size = window_size + 1
    d = t - window_size + 1

    block = trend[d: t + 1] if d >= 0 else -d * [trend[0]] + trend[0: t + 1]
    res = []
    for i in range(window_size - 1):
        res.append(100. * (block[i + 1] - block[i]) / block[i])
    return np.array([res])


def main():
    df = pd.read_csv('../data/datasets/old/GOOGL.csv')

    result_invest = []
    during_test = 1
    during_control = 5
    time_control_by_test = 4
    window_size = 2
    count_state = 0
    times = 250
    times_end = 0
    bagging = 10
    max_learning_size = 25
    start_money = 0
    settings = None

    # for control
    # window, max_learning_size, proportion_test

    for row in range(during_test * (times - 1), during_test * times_end - 1, -during_test):  # 264*3 during_test*10
        # prepare
        during_train = max_learning_size
        end_test = row
        size = len(df)
        # for control
        df0_control = []
        df1_control = []
        for row in range(1, time_control_by_test + 1):
            during_control0 = row * during_test

            df0_control.append(df[(size - end_test - window_size - during_test - during_train - during_control0 - 1): (
                    size - end_test - during_test - during_control0)])
            df1_control.append(df[(size - end_test - during_test - window_size - during_control0 - 1): (
                    size - end_test - during_test - (row - 1) * during_test)])

        # df0_control = df[(size - end_test - window_size - during_test - during_train - during_control - 1): (
        #         size - end_test - during_test - during_control)]
        # df1_control = df[(size - end_test - during_test - window_size - during_control - 1): (
        #         size - end_test - during_test)]
        # for test
        df0_test = df[(size - end_test - window_size - during_test - during_train - 1): (
                size - end_test - during_test)]
        df1_test = df[(size - end_test - window_size - during_test - 1): (size - end_test)]

        data_test = []
        data_test.append(df0_test.Open.values.tolist())
        test = df1_test.Open.values.tolist()
        data_control = []
        control = []
        for row in df0_control:
            data_control.append(row.Open.values.tolist())
        for row in df1_control:
            control.append(row.Open.values.tolist())

        # work
        if settings == None:
            settings = Settings()

        testModel = TestModel(settings, bagging, window_size)
        states_buy, states_sell, total_gains, invest, last_count_state, last_start_money = testModel.run_modeling(
            data_test, test, data_control, control, settings, count_state, start_money)

        settings = testModel.get_last_settings()
        count_state = last_count_state
        start_money = last_start_money
        result_invest.append(invest)

        # fig = plt.figure(figsize=(15, 5))
        # plt.plot(test, color='r', lw=2.)
        # plt.plot(test, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
        # plt.plot(test, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
        # plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
        # plt.legend()
        # plt.show()

    print("all= " + str(result_invest[-1]))


if __name__ == '__main__':
    main()
