from my_framework.strategies import Deep_Evolution_Strategy
import numpy as np
import math


def buy(trend, initial_money, window_size, proportion_test, fee_test, model, count_state=0, start_money=0):
    state = get_state(trend, window_size, window_size)
    current_money = initial_money
    states_sell = []
    states_buy = []

    count = count_state
    if start_money != 0:
        current_money = start_money

    for t in range(window_size, len(trend) - 1):
        action = act(model, state)
        next_state = get_state(trend, t + 1, window_size)

        real_cost = current_money + count * trend[t]

        if action == 1 and current_money >= trend[t]:
            calc_count = real_cost // (trend[t] * proportion_test)

            dem = trend[t] * calc_count * (1 + fee_test)
            if current_money < dem:
                calc_count = current_money // (trend[t] * (1 + fee_test))
            current_money -= trend[t] * calc_count * (1 + fee_test) * 0.9975
            count += calc_count
            states_buy.append(t)
            print('day %d: buy 1 unit at price %f, total balance %f, paper %f' % (
                t, trend[t], current_money + count * trend[t], count))

        elif action == 2 and real_cost >= current_money:
            calc_count = real_cost // (trend[t] * proportion_test)
            current_money += trend[t] * calc_count * (1 + fee_test) * 1.0025
            count -= calc_count
            states_sell.append(t)

            print(
                'day %d, sell 1 unit at price %f, total balance %f, paper %f'
                % (t, trend[t], current_money + count * trend[t], count)
            )
        state = next_state

    holding_sum = count * trend[-1]
    last_money = current_money
    current_money += holding_sum

    invest = ((current_money - initial_money) / initial_money) * 100
    total_gains = current_money - initial_money

    if math.isnan(last_money) or (proportion_test > 10 or proportion_test < 2):
        last_money = -1000000000
        count = 0
        invest = -100
        total_gains = -100

    print('last_money %d, result %f' % (last_money, invest))

    return states_buy, states_sell, total_gains, invest, count, last_money


def get_state(trend, t, window_size):
    window_size = window_size + 1
    d = t - window_size + 1

    block = trend[d:t + 1]
    res = []
    for i in range(window_size - 1):
        res.append(100. * (block[i + 1] - block[i]) / block[i])
    return np.array(res)


def act(model, sequence):
    decision = model.predict(np.array(sequence))
    return np.argmax(decision[0])


class Agent:
    def __init__(self, model, window_size, trend, dif, skip, initial_money
                 , population_size=200, sigma=0.66, learning_rate=0.05
                 , fee_train=0.005, fee_test=0.0005, proportion_train=10, proportion_test=5):
        self.model = model
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.dif = dif
        self.skip = skip
        self.initial_money = initial_money
        self.fee_train = fee_train
        self.fee_test = fee_test
        self.proportion_train = proportion_train
        self.proportion_test = proportion_test

        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            population_size,
            sigma,
            learning_rate
        )

    def act(self, sequence):
        return act(self.model, sequence)
        # decision = self.model.predict(np.array(sequence))
        # return np.argmax(decision[0])

    def get_state(self, t):
        return get_state(self.trend[0], t, self.window_size)

    def get_reward(self, weights):
        initial_money = self.initial_money
        self.model.weights = weights

        holding_sum = 0
        current_money = 0

        for i in range(len(self.trend)):

            state = self.get_state(self.window_size)
            current_money += initial_money
            count = 0

            for t in range(self.window_size, len(self.trend[i]) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1)

                real_cost = current_money + count * self.trend[i][t]

                if action == 1 and current_money >= self.trend[i][t]:
                    calc_count = real_cost // (self.trend[i][t] * self.proportion_train)
                    dem = self.trend[i][t] * calc_count * (1 + self.fee_train)
                    if current_money < dem:
                        calc_count = current_money // (self.trend[i][t] * (1 + self.fee_train))
                    current_money -= self.trend[i][t] * calc_count * (1 + self.fee_train)
                    count += calc_count

                elif action == 2 and real_cost >= current_money:
                    calc_count = real_cost // (self.trend[i][t] * self.proportion_train)
                    current_money += self.trend[i][t] * calc_count * (1 + self.fee_train)
                    count -= calc_count

                state = next_state

            holding_sum += count * self.trend[i][-1]

        result = ((current_money + holding_sum - initial_money) / initial_money) * 100
        return result

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self, close1):
        return buy(close1, self.initial_money, self.window_size, self.proportion_test,
                   self.fee_test, self.get_state, self.act)
