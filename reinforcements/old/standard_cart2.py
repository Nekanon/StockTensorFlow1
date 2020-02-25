# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = list()#deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.25
        self.epsilon_decay = 0.95
        self.learning_rate = 0.005
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
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done, total_ef):
        self.memory.append((state, action, reward, next_state, done, total_ef))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # self.memory = sorted(self.memory, key=lambda column: float(1./column[5]))
        # if len(self.memory) < batch_size*5:
        #     self.memory = self.memory[:batch_size*5]

        minibatch = random.sample(self.memory, batch_size)
        # data prepare
        state_total = None
        target_total = None
        count_local = 0
        for state, action, reward, next_state, done, total_ef in minibatch: #minibatch
            target = self.model.predict(state)
            if state_total is None:
                state_total = state
            else:
                array1 = state_total.tolist()
                array2 = state.tolist()
                state_total = np.append(array1, array2, axis=0)

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
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                if count_local == 0:
                    target_total[0][action] = reward + self.gamma * np.amax(t)
                else:
                    array1 = target_total.tolist()
                    array2 = target[0].tolist()
                    target_total = np.append(array1, [array2], axis=0)
            count_local += 1

        # train
        self.model.fit(state_total, target_total, epochs=5, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1') # CartPole-v1 Pendulum-v0
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n # .shape[0] .n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 100

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        memory_local = []
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            next_state = np.reshape(next_state, [1, state_size])
            # memory_local.append((state, action, reward, next_state, done))
            agent.memorize(state, action, reward, next_state, done, time)
            state = next_state
            if done:
                #transfare memory_local to memory
                # for st, ac, re, nst, d in memory_local:
                #     agent.memorize(state, action, reward, next_state, done, time)
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size and time == 0:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
