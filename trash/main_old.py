import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv('../data/datasets/GOOGL0.csv') #'datasets/GOOGL0.csv'
df2 = pd.read_csv('../data/datasets/GOOGL1.csv')
# df3 = pd.read_csv('datasets/JNJ.csv')
# df = pd.read_csv('datasets/XOM.csv')
# df2 = pd.read_csv('datasets/XOM1.csv')
# df3 = pd.read_csv('datasets/XOM1.csv')

closes = []
closes.append(df.Open.values.tolist())
# closes.append(df3.Close.values.tolist())
close2 = df2.Open.values.tolist()


class Deep_Evolution_Strategy:
    inputs = None

    def __init__(
            self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=100, print_every=1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                        w
                        + self.learning_rate
                        / (self.population_size * self.sigma)
                        * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 200
    SIGMA = 0.66  # 0.15 0.5
    LEARNING_RATE = 0.05  # 0.03 0.15

    def __init__(self, model, window_size, trend, dif, skip, initial_money):
        self.model = model
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.dif = dif
        self.skip = skip
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def get_state(self, t, num_stream=-1):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = []
        if num_stream is -1:
            block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        else:
            block = self.trend[num_stream][d: t + 1] if d >= 0 else -d * [self.trend[num_stream][0]] + self.trend[
                                                                                                           num_stream][
                                                                                                       0: t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(100. * (block[i + 1] - block[i]) / block[i])
        return np.array([res])

    def get_reward(self, weights):
        initial_money = self.initial_money
        self.model.weights = weights

        holding_sum = 0
        starting_money = 0

        for i in range(len(self.trend)):

            state = self.get_state(self.window_size, i)
            starting_money += initial_money
            inventory = []
            quantity = 0
            count = 0

            for t in range(self.window_size, len(self.trend[i]) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1, i)

                real_cost = starting_money + count * self.trend[i][t]

                if action == 1 and starting_money >= self.trend[i][t]:
                    calc_count = 0.1 * real_cost // self.trend[i][t]
                    # if calc_count == 0:
                    #     calc_count = 1
                    # starting_money -= self.trend[i][t] * calc_count
                    starting_money -= self.trend[i][t] * calc_count * 1.005
                    count += calc_count

                elif action == 2 and real_cost >= starting_money:
                    calc_count = 0.1 * real_cost // self.trend[i][t]
                    # if calc_count == 0:
                    #     calc_count = 1
                    # starting_money += self.trend[i][t] * calc_count
                    starting_money += self.trend[i][t] * calc_count * 1.005
                    count -= calc_count

                state = next_state

            holding_sum += count * self.trend[i][-1]

        result = ((starting_money + holding_sum - initial_money) / initial_money) * 100
        return result

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self, close1):
        self.trend = close1
        initial_money = self.initial_money
        state = self.get_state(self.window_size)
        starting_money = initial_money
        states_sell = []
        states_buy = []

        count = 0

        for t in range(self.window_size, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            real_cost = starting_money + count * self.trend[t]

            if action == 1 and starting_money >= self.trend[t]:
                calc_count = 0.2 * real_cost // self.trend[t]
                # if calc_count == 0:
                #     calc_count = 1
                starting_money -= close2[t] * calc_count * 1.0005
                # starting_money -= self.trend[t] * calc_count * 0.0005
                count += calc_count
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f, paper %f' % (
                    t, self.trend[t], starting_money + count * self.trend[t], count))

            elif action == 2 and real_cost >= starting_money:
                calc_count = 0.2 * real_cost // self.trend[t]
                # if calc_count == 0:
                #     calc_count = 1
                starting_money += self.trend[t] * calc_count * 1.0005
                # starting_money -= close2[t] * calc_count * 0.0005
                count -= calc_count
                states_sell.append(t)

                print(
                    'day %d, sell 1 unit at price %f, total balance %f, paper %f'
                    % (t, self.trend[t], starting_money + count * self.trend[t], count)
                )
            state = next_state

        holding_sum = count * self.trend[-1]
        starting_money += holding_sum

        invest = ((starting_money - initial_money) / initial_money) * 100
        total_gains = starting_money - initial_money
        return states_buy, states_sell, total_gains, invest

def getCount(arr_arr, co):
    res_count = 0
    for row_arr in arr_arr:
        if containsArray(row_arr, co):
            res_count += 1
    return res_count

def containsArray(arr, co):
    for row in arr:
        if row == co:
            return True
    return False

window_size = 5
skip = 1
initial_money = 100000

invest_array = []
states_buy_array = []
states_sell_array = []
iterations = 10


for iter in range(iterations):
    model = Model(input_size=window_size, layer_size=2, output_size=3)
    agent = Agent(model=model,
          window_size=window_size,
          trend=closes,
          dif=closes,
          skip=skip,
          initial_money=initial_money)
    agent.fit(iterations=400, checkpoint=50)

    states_buy, states_sell, total_gains, invest = agent.buy(close1=close2)
    states_buy_array.append(states_buy)
    states_sell_array.append(states_sell)
    invest_array.append(invest)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(close2, color='r', lw=2.)
    plt.plot(close2, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(close2, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
    plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
    plt.legend()
    plt.show()

mean1 = statistics.mean(invest_array)
std1 = statistics.stdev(invest_array)
print('mean %f, std %f, prof %f' % (round(mean1, 2), round(std1, 2), round(mean1/std1, 2)))

result_status = []
for row in range(5, 35):
    count_buy = getCount(states_buy_array, row)
    count_sell = getCount(states_sell_array, row)
    count_none = iterations - count_buy - count_sell
    if count_buy > count_sell and count_buy > count_none:
        result_status.append('+')
    elif count_sell > count_buy and count_sell > count_none:
        result_status.append('-')
    else:
        result_status.append('0')



print(states_buy_array)
print(states_sell_array)
print(result_status)

# import utils.autoencoder as ae
#
# data1 = []
# window = 10
# length = len(closes[0]) - window
# for row in range(window):
#     start = row
#     end = length + row
#     data1.append([closes[0][i] for i in range(start, end)])
# data1 = pd.DataFrame(data1)
# # data1 = data1.transpose()
# vec = ae.reducedimension(input_=data1, dimension=5, learning_rate=0.01, hidden_layer=20, epoch=100000)
# re = 4
