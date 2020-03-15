import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

df = pd.read_csv('../data/datasets/GOOGL0.csv')
df2 = pd.read_csv('../data/datasets/GOOGL1.csv')

closes = []
closes.append(df.Close.values.tolist())
close2 = df2.Close.values.tolist()


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


class Model2L:
    def __init__(self, input_size, layer_size, output_size, layer2_size=5):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, layer2_size),
            np.random.randn(layer2_size, output_size),
            np.random.randn(1, layer2_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed1 = np.dot(inputs, self.weights[0]) + self.weights[-1]
        feed2 = np.dot(feed1, self.weights[1]) + self.weights[-2]
        decision = np.dot(feed2, self.weights[2])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 100
    SIGMA = 0.5  # 0.15 0.5
    LEARNING_RATE = 0.1  # 0.03 0.15

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

            state = self.get_state(0, i)
            starting_money += initial_money
            inventory = []
            quantity = 0
            count = 0

            for t in range(0, len(self.trend[i]) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1, i)

                real_cost = starting_money + count * self.trend[i][t]

                if action == 1 and starting_money >= self.trend[i][t]:
                    calc_count = 0.2 * real_cost // self.trend[i][t]
                    if calc_count == 0:
                        calc_count = 1
                    starting_money -= self.trend[i][t] * calc_count * 1.005
                    count += calc_count

                elif action == 2 and real_cost >= starting_money:
                    calc_count = 0.2 * real_cost // self.trend[i][t]
                    if calc_count == 0:
                        calc_count = 1
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

        for t in range(self.window_size, len(close2) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            real_cost = initial_money + count * close2[t]

            if action == 1 and initial_money >= close2[t]:
                calc_count = 0.2 * real_cost // close2[t]
                if calc_count == 0:
                    calc_count = 1
                initial_money -= close2[t] * calc_count
                initial_money -= self.trend[t] * calc_count * 0.00025
                count += calc_count
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f, paper %f' % (
                    t, self.trend[t], initial_money + count * close2[t], count))

            elif action == 2 and real_cost >= initial_money:
                calc_count = 0.2 * real_cost // close2[t]
                if calc_count == 0:
                    calc_count = 1
                initial_money += close2[t] * calc_count
                initial_money -= close2[t] * calc_count * 0.00025
                count -= calc_count
                states_sell.append(t)

                print(
                    'day %d, sell 1 unit at price %f, total balance %f, paper %f'
                    % (t, close2[t], initial_money + count * close2[t], count)
                )
            state = next_state

        holding_sum = count * close2[-1]
        initial_money += holding_sum

        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest


window_size = 10
skip = 1
initial_money = 3000

model = Model(input_size=window_size, layer_size=6, output_size=3)
agent = Agent(model=model,
              window_size=window_size,
              trend=closes,
              dif=closes,
              skip=skip,
              initial_money=initial_money)
agent.fit(iterations=500, checkpoint=10)

states_buy, states_sell, total_gains, invest = agent.buy(close1=close2)

fig = plt.figure(figsize=(30, 10))
plt.plot(close2, color='r', lw=2.)
plt.plot(close2, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
plt.plot(close2, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
plt.legend()
plt.show()
