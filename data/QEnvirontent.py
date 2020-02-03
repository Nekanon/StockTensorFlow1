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

    def step(self, action):  # -1, 0 ,1
        done = False
        if self.count > 700:
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

    def steps(self):
        pass

    def test(self, model):

        test_init_cost = self.test_y[0] * 4 * self.fee
        test_all_free_money = test_init_cost

        test_count_stock = 0
        size = self.test_y.shape[0]
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
        test_result = round(float(test_all_free_money / test_init_cost - 1), 4)
        return test_result
