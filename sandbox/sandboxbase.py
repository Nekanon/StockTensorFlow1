class SandBoxBase:
    def __init__(self, model, data_in=None, data_out=None):
        self.input_data = data_in
        self.output_data = data_out
        self.model = model

    def setDataIn(self, data_in):
        self.input_data = data_in

    def setDataOut(self, data_out):
        self.output_data = data_out

    def setModel(self, model):
        self.model = model

    def getModel(self):
        return self.model

    def _retrain_model(self):
        pass

    def run(self):
        pass


class SandBoxTimeSeries(SandBoxBase):
    def __init__(self, model, data_in=None, data_out=None, step_retrain=1, init_cash=10000):
        super().__init__(model=model,
                         data_in=data_in,
                         data_out=data_out)
        self.step_retrain = step_retrain
        self.init_cash = init_cash

    def _retrain_model(self):
        pass

    def get_action(self):
        return 0

    def get_reward(self, action):
        return 1.0

    def getCurrentPrice(self, i):
        return 1.0

    def run(self):
        # steps
        start_sand = 0
        end_sand = 100

        current_cash = self.init_cash
        total_cost = self.init_cash
        current_count = 0

        for i_epoch in range(start_sand, end_sand):
            # retrain
            if i_epoch % self.step_retrain == 0:
                self._retrain_model()

            # get action
            action = self.get_action()

            # get environment reaction
            reward = self.get_reward(action)

            # total cost
            total_cost = current_cash + current_count * self.getCurrentPrice(i_epoch)

        total_eff = (total_cost / self.init_cash - 1.0) * 100.0
