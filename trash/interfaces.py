#
class DataDescription:
    def __init__(self, streams=[], last=None):
        self.streams = streams
        self.last = last

    def getStreams(self):
        return self.streams


class ValidationStrategy:
    def __init__(self):
        pass


class DataSet:
    def __init__(self, dataDescription, start=0, end=-1, validationStrategy=None):
        self.dataDescription = dataDescription
        self.start = start
        self.end = end
        self.validationStrategy = validationStrategy

    def getData(tiker="GOOGL", window=10, des3=False, tiker_test=None):
        pass

    def getVolume(dataFrame, drop_range=0, rangevolume=10):
        pass


#
class Environment(DataSet):
    def __init__(self, dataSet=None):
        self.dataSet = dataSet

    def __init__(self, data_y, data_x, test_y=None, test_x=None):
        pass

    def getState(self):
        pass

    def step(self, action):
        pass

    def getReward(self, action):
        pass

    def size(self):
        return 0


#
class Model:
    def __init__(self, input_shape, layer_architecture, output_shape):
        pass

    def predict(self, inputs):
        pass

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self):
        pass

    def load(self):
        pass


class SandBox:
    def __init__(self, dataSet, agent, periodRetrain=1):
        self.dataSet = dataSet
        self.agent = agent
        self.periodRetrain = periodRetrain

    def setModel(self):
        pass

    def getModel(self):
        pass

    def runSimulate(self):
        pass

    def getLastResult(self):
        pass


#
class StrategyLearn:
    def __init__(self):
        pass

    def train(self, epoch):
        pass

    def getWeights(self):
        pass


#
class LearnerAgent:
    def __init__(self, model, dataSet, strategyLearn, window_size=1, population=10, sigma=0.5, learning_rate=0.01):
        self.model = model
        self.dataSet = dataSet
        self.strategyLearn = strategyLearn
        self.window_size = window_size
        self.population = population
        self.sigma = sigma
        self.learning_rate = learning_rate

    def __init__(self, model, strategyLearn, window_size=1, population=10, sigma=0.5, learning_rate=0.01):
        self.model = model
        self.dataSet = dataSet
        self.strategyLearn = strategyLearn
        self.window_size = window_size
        self.population = population
        self.sigma = sigma
        self.learning_rate = learning_rate

    def act(self):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass

    def fit(self):
        pass

    def test(self):
        pass


# use case
# prepare data
dataDescription = DataDescription([
    ("GOOGL", "dif", "Close", "Open", 0),
    ("GOOGL", "dif", "High", "Open", 0),
    ("GOOGL", "dif", "Low", "Open", 0),
    ("GOOGL", "dif", "Close", "Close", 1),
    ("GOOGL", "AM", "Volume", 10)
])
validationStrategy = ValidationStrategy()
dataSet = DataSet(dataDescription, validationStrategy=validationStrategy)

# prepare strategy
strategyLearn = StrategyLearn()

# prepare agent for one fit
model = Model(input_shape=(5, 10), layer_architecture=None, output_shape=(3, 1))
learnerAgentOne = LearnerAgent(model=model, dataSet=dataSet, strategyLearn=strategyLearn)
learnerAgentOne.fit()

# prepare agent for multi fit
learnerAgentMulti = LearnerAgent(model=model, strategyLearn=strategyLearn)
sandBox = SandBox(dataSet=dataSet, periodRetrain=5, agent=learnerAgentMulti)
sandBox.runSimulate()
result = sandBox.getLastResult()
