import numpy as np


class QEnvironment:
    def __init__(self, data_y, data_x, test_y=None, test_x=None):
        self.data_y = data_y
        self.data_x = data_x
        self.test_y = test_y
        self.test_x = test_x

        self.done = False
        self.count = 0
        self.count_stock = 0
        self.fee = 1.001
        self.init_cost = self.data_y[0] * 4 * self.fee  # 10000
        self.all_free_money = self.init_cost

    def getSizes(self):
        state_size = 1
        action_size = 3
        return

    def getState(self):
        return self.data_x[self.count]

    def getStates(self):
        return self.data_x

    def step(self, action):  # -1, 0 ,1
        done = False
        if self.count >= self.size() - 3:
            done = True
        if self.count >= len(self.data_y) - 1:
            self.done = True

        all_cost_start = self.all_free_money + self.count_stock * self.data_y[self.count]
        cost_start = round(float((all_cost_start / self.init_cost - 1)), 4)

        if action == 0 and self.all_free_money > self.data_y[self.count] * self.fee:
            buy_count = 2  # int(self.all_free_money // self.data_y[self.count] * self.fee * 0.2)
            self.all_free_money -= buy_count * self.data_y[self.count] * self.fee
            self.count_stock += buy_count
        elif action == 1 and self.count_stock > 0:
            sell_count = 2  # int(self.count * 0.5)
            self.all_free_money += self.data_y[self.count] * (1 / self.fee) * sell_count
            self.count_stock -= sell_count
        else:
            pass

        self.count += 1
        all_cost_end = self.all_free_money + self.count_stock * self.data_y[self.count]
        state = self.getState()
        cost_end = round(float((all_cost_end / self.init_cost - 1)), 4)
        reward = round(all_cost_end / all_cost_start - 1, 4)

        if done:
            self.count_stock = 0
            self.count = 0
            self.all_free_money = self.init_cost

        return state, reward, done

    def steps(self, agent):
        done = False
        state = self.getState()
        profit = 1.0
        while not done:
            action = agent.act(state)
            next_state, reward, done = self.step(action)
            profit *= 1 + reward
            state = next_state
        return profit

    def getrewards(self, actions):

        all_free_money = self.init_cost
        count_stock = 0
        rewards = []

        for index, action in enumerate(actions):
            if index == len(actions)-1:
                break
            all_cost_start = all_free_money + count_stock * self.data_y[index]
            # cost_start = round(float((all_cost_start / self.init_cost)), 4)

            if action == 0 and all_free_money > self.data_y[index] * self.fee:
                buy_count = 2
                all_free_money -= buy_count * self.data_y[index] * self.fee
                count_stock += buy_count
            elif action == 1 and self.count_stock > 0:
                sell_count = 2  # int(self.count * 0.5)
                all_free_money += self.data_y[index] * (1 / self.fee) * sell_count
                count_stock -= sell_count
            else:
                pass

            all_cost_end = all_free_money + count_stock * self.data_y[index+1]
            # cost_end = round(float((all_cost_end / self.init_cost)), 4)

            reward = round(all_cost_end / all_cost_start - 1, 4)
            rewards.append(reward)

        return rewards

    def stepsnew(self, agent):
        done = False
        states = self.getStates()
        actions = agent.acts(states)
        profit = 1.0
        rewards = self.getrewards(actions)
        for reward in rewards:
            profit *= 1 + reward
        return profit

    def test(self, model):

        test_init_cost = self.test_y[0] * 4 * self.fee
        test_all_free_money = test_init_cost

        test_count_stock = 0
        size = self.test_y.shape[0]-1
        for index in range(size):
            state = self.test_x[index]
            state = np.array([state.tolist()])
            action = model.predict(state)[0]
            action = np.argmax(action)

            if action == 0 and test_all_free_money > self.test_y[index] * self.fee:
                buy_count = 2
                test_all_free_money -= buy_count * self.test_y[index] * self.fee
                test_count_stock += buy_count
            elif action == 1 and test_count_stock > 0:
                sell_count = 2
                test_all_free_money += self.test_y[index] * (1 / self.fee) * sell_count
                test_count_stock -= sell_count
            else:
                pass

        test_all_free_money += self.test_y[size - 1] * test_count_stock * (1 / self.fee)
        test_result = round(float(test_all_free_money / test_init_cost ), 4)
        return test_result

    def testnew(self, agent):

        test_init_cost = self.test_y[0] * 5 * self.fee
        test_all_free_money = test_init_cost

        test_count_stock = 0
        size = self.test_y.shape[0]

        states = np.array(self.test_x.tolist())
        actions = agent.acts(states)

        for index, action in enumerate(actions):
            # state = self.test_x[index]
            # state = np.array([state.tolist()])
            # action = model.predict(state)[0]
            # action = np.argmax(action)

            if action == 0 and test_all_free_money > self.test_y[index] * self.fee:
                buy_count = 2
                test_all_free_money -= buy_count * self.test_y[index] * self.fee
                test_count_stock += buy_count
            elif action == 1 and test_count_stock > 0:
                sell_count = 2
                test_all_free_money += self.test_y[index] * (1 / self.fee) * sell_count
                test_count_stock -= sell_count
            else:
                pass

        test_all_free_money += self.test_y[size - 1] * test_count_stock * (1 / self.fee)
        test_result = round(float(test_all_free_money / test_init_cost ), 4)
        return test_result


    def size(self):
        return len(self.data_y)