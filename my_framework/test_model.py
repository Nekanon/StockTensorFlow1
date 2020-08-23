from my_framework.agents import *
from my_framework.models import Model, BaggingModel


def minim(func, start_array_settings):
    best_array_settings = start_array_settings
    best_res = -100.0
    best_count = 0
    best_max_count = 2

    step_array_settings = start_array_settings
    base_res = func(step_array_settings)

    grad = []
    for row1 in range(len(step_array_settings)):
        for row2 in range(-2, 3):
            sub_step_array_settings = step_array_settings.copy()
            sub_step_array_settings[row1] += row2
            res = func(sub_step_array_settings)

            grad.append({'pos': sub_step_array_settings, 'value': res - base_res})

    grad = sorted(grad, key=lambda k: k['value'])
    best_array_settings = grad[-1]['pos']

    print("settings=" + str(grad[-1]['pos']))
    print("base_res=" + str(grad[-1]['value'] + base_res))
    return best_array_settings


class Settings:
    def __init__(self, population_size=50, sigma=0.5, learning_rate=0.01, fee_train=0.0005, fee_test=0.0005, #0.65 0.01
                 proportion_train=20, proportion_test=0.5, window_size=2, iterations_test=20, learning_size=5,
                 mud_learning_size=2, max_learning_size=66):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.fee_train = fee_train
        self.fee_test = fee_test
        self.proportion_train = proportion_train
        self.proportion_test = proportion_test
        self.window_size = window_size
        self.iterations_test = iterations_test
        self.learning_size = learning_size
        self.mud_learning_size = mud_learning_size
        self.max_learning_size = max_learning_size

    def get(self):
        return [
            # self.window_size,
            # self.proportion_test,
            self.mud_learning_size
        ]

    def set(self, array):
        # self.window_size = array[0]
        # self.proportion_test = array[0]
        self.mud_learning_size = array[0]


class TestModel:
    def __init__(self, settings, iterations=5, window_size=5):
        self.iterations = iterations
        self.settings = settings
        self.window_size = window_size
        self.initial_money = 1000000
        self.test = []
        self.data_test = []

    def run_test(self, data_test, test, settings, count_state, start_money):
        need_size = len(data_test[0]) - self.window_size - settings.mud_learning_size * settings.learning_size
        data_test_new = [data_test[0][need_size:]]

        if settings.window_size < 1:
            settings.window_size = 1
        if settings.mud_learning_size < 1:
            settings.mud_learning_size = 1
        if settings.mud_learning_size > 5:
            settings.mud_learning_size = 5
        if settings.proportion_test < 2:
            settings.proportion_test = 2
        if settings.proportion_test > 4:
            settings.proportion_test = 4

        bagging_model = BaggingModel()
        for iteration in range(self.iterations):
            model = Model(input_size=settings.window_size, layer_size=2, output_size=3)
            agent = Agent(model=model,
                          window_size=settings.window_size,
                          trend=data_test_new,
                          dif=data_test_new,
                          skip=1,
                          initial_money=self.initial_money,
                          population_size=settings.population_size,  # 200 50
                          sigma=settings.sigma,  # 0.66 0.5
                          learning_rate=settings.learning_rate,  # 0.05 0.025
                          fee_train=settings.fee_train,  # 0.005
                          fee_test=settings.fee_test,  # .0005
                          proportion_train=settings.proportion_train,  # 0.1 0.05
                          proportion_test=settings.proportion_test)  # 0.2
            agent.fit(iterations=settings.iterations_test, checkpoint=settings.iterations_test)  # 500 1000
            bagging_model.append(model)

        # test = test[5-settings.window_size:]

        return buy(test, self.initial_money, settings.window_size, settings.proportion_test, settings.fee_test,
                   bagging_model, count_state, start_money)

    def run_test1(self, array_settings):
        settings = Settings()
        settings.set(array_settings)

        if array_settings[0]>5 or array_settings[0]<2:# or array_settings[1]<1 or array_settings[1]>10:
            return -1

        resAll = 1
        for control_count in range(len(self.data_test)):

            control_data = []
            need_size = len(
                self.data_test[control_count]) - self.window_size - settings.mud_learning_size * settings.learning_size
            control_data.append(self.data_test[control_count][need_size:])
            control = self.test[control_count]

            bagging_model = BaggingModel()
            for iteration in range(self.iterations):
                model = Model(input_size=self.window_size, layer_size=2, output_size=3)
                agent = Agent(model=model,
                              window_size=self.window_size,
                              trend=control_data,
                              dif=control_data,
                              skip=1,
                              initial_money=self.initial_money,
                              population_size=settings.population_size,  # 200 50
                              sigma=settings.sigma,  # 0.66 0.5
                              learning_rate=settings.learning_rate,  # 0.05 0.025
                              fee_train=settings.fee_train,  # 0.005
                              fee_test=settings.fee_test,  # .0005
                              proportion_train=settings.proportion_train,  # 0.1 0.05
                              proportion_test=settings.proportion_test)  # 0.2
                agent.fit(iterations=settings.iterations_test, checkpoint=300)  # 500 1000
                bagging_model.append(model)

            res = buy(control, self.initial_money, self.window_size, settings.proportion_test, settings.fee_test,
                      bagging_model)
            resAll *= 1 + res[3] / 100
        print("--- result=" + str((resAll - 1) * 100.0))
        return resAll

    def search_metaparams(self, data_control, control, settings):
        self.test = control
        self.data_test = data_control
        array_settings = np.array(settings.get())

        # optimize
        array_settings = minim(self.run_test1, array_settings)

        sett = Settings(iterations_test=300)
        sett.set(array_settings)
        return sett

    def run_modeling(self, data_test, test, data_control, control, settings, count_state, initial_money):
        self.data_control = data_control
        self.control = control

        # search metaparams by control data
        self.settings = settings #self.search_metaparams(data_control, control, settings)

        res = self.run_test(data_test, test, self.settings, count_state, initial_money)
        return res

    def get_last_settings(self):
        return self.settings
