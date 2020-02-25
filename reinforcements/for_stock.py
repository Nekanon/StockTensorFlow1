# -*- coding: utf-8 -*-
import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf

import copy

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size, input_model=None, front_model=None, size_deque=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=size_deque)
        self.gamma = 0.8  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.8
        self.learning_rate = 0.1
        if input_model is not None:
            self.model = input_model
            self.target_model = input_model
        else:
            self.model = self._build_model(front_model)
            self.target_model = self._build_model(front_model)
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, front_model=None):

        input_front = None
        if front_model is None:
            input_front = Input(shape=(self.state_size[0], self.state_size[1], 1))
            flat_front = Flatten()(input_front)
            x1 = Dense(15)(flat_front)
            front_model = Model(input_front, x1, name="front")
        else:
            input_front = front_model.input

        # x = Dense(10)(x)
        input_back = Input(shape=(15,))
        x2 = Dense(5)(input_back)
        x2 = Dense(3)(x2)
        back_model = Model(input_back, x2, name="back")

        # qlearner = Dense(self.action_size, activation='linear')(x)
        # qlearner = Model(input_img, qlearner, name="qlearner")
        qlearner = Model(input_front, back_model(front_model(input_front)), name="full_qlearner")

        qlearner.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return qlearner

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array([state.tolist()])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # data prepare
        minibatch = random.sample(self.memory, batch_size)
        state_total = None
        target_total = None
        count_local = 0

        for state, action, reward, next_state, done in minibatch:
            need_state = np.array([state])
            target = self.model.predict(need_state)
            if state_total is None:
                ss = state.tolist()
                state_total = np.array([ss])
            else:
                array1 = state_total.tolist()
                array2 = state.tolist()
                state_total = np.append(array1, [array2], axis=0)

            if target_total is None:
                target_total = target
            if done:
                target[0][action] = reward
                if count_local == 0:
                    target_total[0][action] = reward
                else:
                    array1 = target_total.tolist()
                    array2 = target[0].tolist()
                    target_total = np.append(array1, [array2], axis=0)
            else:
                need_state = np.array([next_state])
                t = self.target_model.predict(need_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                if count_local == 0:
                    target_total[0][action] = reward + self.gamma * np.amax(t)
                else:
                    array1 = target_total.tolist()
                    array2 = target[0].tolist()
                    target_total = np.append(array1, [array2], axis=0)
            count_local += 1

        # train
        self.model.fit(state_total, target_total, epochs=1, verbose=0)  # 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class GeneticAgent:
    def __init__(self, environment, amplitude_wights=1.0, front_model=None):
        self.environment = environment
        self.front_model = front_model
        self.model = self._build_model(front_model)
        self.assessment = 0.0
        self.environment_current = environment
        self.mutation_rate_max = 0.5
        self.mutation_rate_min = 0.05

    def getModel(self):
        return self.model

    def _build_model(self, front_model=None):

        input_front = None
        if front_model is None:
            input_front = Input(shape=(6, 10, 1))
            flat_front = Flatten()(input_front)
            x1 = Dense(20)(flat_front)
            front_model = Model(input_front, x1, name="front")
        else:
            input_front = front_model.input

        input_back = Input(shape=(15,))
        x2 = Dense(8)(input_back)
        x2 = Dense(3)(x2)
        back_model = Model(input_back, x2, name="back")

        qlearner = Model(input_front, back_model(front_model(input_front)), name="full_qlearner")
        weights = qlearner.get_weights()

        qlearner.compile(loss="mean_squared_error", optimizer="adam")
        return qlearner

    def myrandomRange(self, rate):
        int1 = int(rate * 1000)
        return float(random.randrange(-int1, int1) / 1000.0)

    def doMutant(self):
        result = None
        mutation_rate_current = self.mutation_rate_min

        layers = self.model.get_weights()

        llayers = []
        for index_layer, layer in enumerate(layers):
            llayer = layer.tolist()
            for index_neur, neur in enumerate(llayer):
                lneur = neur
                if index_layer % 2:
                    llayer[index_neur] += self.myrandomRange(mutation_rate_current)
                else:
                    for index_link, link in enumerate(neur):
                        llayer[index_neur][index_link] += self.myrandomRange(mutation_rate_current)

            llayers.append(np.array(llayer))
        layers = np.array(llayers)

        result = self._build_model(self.front_model)
        result.set_weights(layers)

        return result

    def breading(self):
        pass

    def doAssessment(self):
        # self.environment_current = self.environment.steps(self)
        return self.environment.stepsnew(self)

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        state = np.array([state.tolist()])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def acts(self, states):
        size = len(states.tolist())
        acts_values = self.model.predict(x=states)
        actions = [np.argmax(i) for i in acts_values]
        return actions

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def gsModel(self, model):
        self.model = model
        return self.model


class QStock:
    def __init__(self, environment, data, front_model=None):
        self.environment = environment
        self.data = data
        state_size = (self.data.shape[1], self.data.shape[2])
        action_size = 3
        self.agent = DQNAgent(state_size, action_size, front_model=front_model, size_deque=self.environment.size())

    def run(self):
        env = self.environment
        # shape = self.data.shape
        # state_size = (shape[1], shape[2])
        # action_size = 3

        agent = self.agent  # DQNAgent(state_size, action_size)
        batch_size = env.size() - 1
        for e in range(EPISODES):

            profit = 1
            state = env.getState()
            size_pole = env.size()
            for time in range(size_pole):

                action = agent.act(state)
                next_state, reward, done = env.step(action)
                profit *= 1 + reward
                agent.memorize(state, action, reward * 100, next_state, done)
                state = next_state
                if done:
                    test_result = env.test(agent.model)
                    agent.update_target_model()
                    print("episode: {}/{}, score: {:.3}, test_score: {:.3}, e: {:.2}"
                          .format(e, EPISODES, profit, test_result, agent.epsilon))
                    break
                if len(agent.memory) > batch_size and time == 0:
                    agent.replay(batch_size)

            # if e % 40 == 0 and e != 0:
            #     agent.learning_rate /= 2
            #     agent.model.compile(loss=agent._huber_loss, optimizer=Adam(lr=agent.learning_rate))


class QGeneticStock:
    def __init__(self, environment, data, front_model=None):
        self.environment = environment
        self.data = data
        self.front_model = front_model
        state_size = (self.data.shape[1], self.data.shape[2])
        action_size = 3
        self.population = 30
        self.agents = []
        for row in range(self.population):
            agent = GeneticAgent(environment=environment, amplitude_wights=1.0, front_model=front_model)
            self.agents.append({"agent": agent, "assessment": 0.0})

    def run(self):

        # assessment
        for row in self.agents:
            row["assessment"] = row["agent"].doAssessment()

        for e in range(EPISODES):

            # mutation
            mutants = []
            for row in self.agents:
                mutante_model = row["agent"].doMutant()
                mutante = GeneticAgent(environment=self.environment, amplitude_wights=1.0, front_model=self.front_model)
                mutante.gsModel(mutante_model)
                post_mutant = row["agent"].doAssessment()
                mutants.append({"agent": mutante, "assessment": mutante.doAssessment()})

            # #breeding
            # for row in self.agents:
            #     row["agent"].breeding()

            # its life
            # self.agents = self.agents + mutants
            # self.agents.sort(key=lambda kv: (kv[1], kv[0]))
            # self.agents = sorted(self.agents, key=lambda i: -i["assessment"])
            # self.agents = self.agents[:self.population]

            print("-----------------------------------")
            best_agent = self.agents[0]
            test_result = self.environment.testnew(best_agent["agent"])
            print("episode: %s", e)
            print("score: %s", best_agent["assessment"])
            print("test: %s", test_result)
