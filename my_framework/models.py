import numpy as np


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

    def pre_predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        return feed

    def get_area(self, data):
        result = []
        x = []
        y = []
        for row in data:
            narray1 = self.pre_predict(row)[0].tolist()
            result.append(np.argmax(self.predict(row)))
            x.append(narray1[0])
            y.append(narray1[1])
        return result, x, y


def act(model, sequence):
    decision = model.predict(np.array(sequence))
    return np.argmax(decision[0])

class BaggingModel:
    def __init__(self):
        self.models = []

    def models(self, models=None):
        if models is not None:
            self.models = models
        return self.models

    def append(self, model):
        self.models.append(model)


    def predict(self, input):
        actions = []
        for model in self.models:
            # result = model.predict(input)
            # actions.append(np.argmax(result[0]))
            actions.append(act(model, input))
        return self.__gol(actions)

    def __gol(self, actions):
        result2 = {}
        for row in actions:
            value = 0
            try:
                value = result2[row]
            except:
                pass
            result2[row] = value + 1
        result3 = [0, 0, 0]
        for k, v in result2.items():
            result3[k] = v

        if result3[1] == result3[2]:
            result3[0] += 1000
        if result3[0] == result3[1] and result3[1] > result3[2]:
            result3[1] += 1000
        if result3[0] == result3[2] and result3[2] > result3[1]:
            result3[2] += 1000
        return np.array([result3])