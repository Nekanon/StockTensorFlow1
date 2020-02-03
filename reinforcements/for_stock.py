# -*- coding: utf-8 -*-
import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size, input_model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        if input_model is not None:
            self.model = input_model
            self.target_model = input_model
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
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

    def _build_model(self):
        input_img = Input(shape=(self.state_size[0], self.state_size[1], 1))
        flat_img = Flatten()(input_img)
        x = Dense(20)(flat_img)
        x = Dense(10)(x)

        qlearner = Dense(self.action_size, activation='linear')(x)
        qlearner = Model(input_img, qlearner, name="qlearner")
        qlearner.compile(loss=self._huber_loss,
                         optimizer=Adam(lr=self.learning_rate))
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
        self.model.fit(state_total, target_total, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class QStock:
    def __init__(self, environment, data):
        self.environment = environment
        self.data = data

    def run(self):
        env = self.environment
        shape = self.data.shape

        state_size = (shape[1], shape[2])
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        batch_size = 1000
        for e in range(EPISODES):

            profit = 1
            state = env.getState()
            for time in range(1000):

                action = agent.act(state)
                next_state, reward, done = env.step(action)
                profit *= 1 + reward
                agent.memorize(state, action, reward*100, next_state, done)
                state = next_state
                if done:
                    test_result = env.test(agent.model)
                    agent.update_target_model()
                    print("episode: {}/{}, score: {:.3}, test_score: {:.3}, e: {:.2}"
                          .format(e, EPISODES, profit, test_result, agent.epsilon))
                    break
                if len(agent.memory) > batch_size and time == 0:
                    agent.replay(batch_size)
