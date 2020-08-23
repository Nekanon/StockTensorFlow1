import numpy as np
import time


class Deep_Evolution_Strategy:
    inputs = None

    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = int(population_size)
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def _add_matrix(self, X, Y):
        result = X.copy()
        for i in range(len(X)):
            # iterate through columns
            for j in range(len(X[0])):
                result[i] = X[i].__add__(Y[i])
        return result

    def get_weights(self):
        return self.weights

    def _train_new(self, epoch=100):

        weights = [self.weights]

        for i in range(int(epoch)):
            population = []
            arr1 = []
            rewards = np.zeros(self.population_size + 1)

            for k in range(self.population_size + 1):
                x = []
                if k == self.population_size:
                    for w in self.weights:
                        x.append(np.zeros(w.shape))
                else:
                    for w in self.weights:
                        x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size + 1):
                if k == self.population_size:
                    weights_population = self.weights
                else:
                    weights_population = self._get_weight_from_population(
                        self.weights, population[k]
                    )
                rewards[k] = self.reward_function(weights_population)
                arr1.append({'weight': weights_population, 'reward': rewards[k]})

            arr1.sort(key=lambda k: k['reward'], reverse=True)
            self.weights = arr1[0]['weight']
            weights = arr1[:10]
            re = 4

    def _train_old(self, epoch=100, print_every=100):
        lasttime = time.time()
        for i in range(int(epoch)):
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

    def train(self, epoch=100, print_every=1):
        return self._train_new(epoch)
        # return self._train_old(epoch, print_every)
