class QEnvironment:
    def __init__(self, data_y, data_x, test_y=None):
        self.data_y = data_y
        self.data_x = data_x
        self.test_y = test_y

        self.done = False
        self.count = 0
        self.count_stock = 0
        self.fee = 1.001
        self.init_cost = self.data_y[0]*4*self.fee #10000
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

        if action == 0 and self.all_free_money > self.data_y[self.count] * self.fee:
            buy_count = 2 #int(self.all_free_money // self.data_y[self.count] * self.fee * 0.2)
            self.all_free_money -= buy_count * self.data_y[self.count] * self.fee
            self.count_stock += buy_count
        elif action == 1 and self.count_stock > 0:
            sell_count = 2 #int(self.count * 0.5)
            self.all_free_money += self.data_y[self.count] * ( 1 / self.fee) * sell_count
            self.count_stock -= sell_count
        else:
            pass

        all_cost = self.all_free_money + self.count_stock * self.data_y[self.count]
        self.count += 1
        state = self.getState()
        reward = round(float((all_cost / self.init_cost - 1)), 4)

        if done:
            self.count_stock = 0
            self.count = 0
            self.all_free_money = self.init_cost


        return state, reward, done

    def steps(self):
        pass
