from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, \
    UpSampling1D, UpSampling2D, Lambda, Dropout
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import L1L2
import random

window = 10


# def getData(window=10, des3=False):
#     tt = "Low"
#     tt1 = "Low"
#     # train
#     df = pd.read_csv('../data/datasets/GOOGL0.csv')
#     df = (df[tt].diff() * 100 / df[tt1]).dropna().values.tolist()
#     # window = 10
#     train_data = []
#     for row in range(window, len(df)):
#         start = row - window
#         end = row
#         # noise = np.random.normal(0, 1, end-start).tolist()
#         train_data.append([df[i] for i in range(start, end)])
#     train_data = np.array(train_data)
#     train_data = train_data.astype('float32')
#     if des3:
#         train_data = np.reshape(train_data, (len(train_data), window, 1))
#     else:
#         train_data = np.reshape(train_data, (len(train_data), window))
#
#     # test
#     df1 = pd.read_csv('../data/datasets/GOOGL0.csv')
#     df1 = (df1[tt].diff() * 100 / df1[tt1]).dropna().values.tolist()
#     test_data = []
#     for row in range(window, len(df1)):
#         start = row - window
#         end = row
#         test_data.append([df1[i] for i in range(start, end)])
#     test_data = np.array(test_data)
#     test_data = test_data.astype('float32')
#     if des3:
#         test_data = np.reshape(test_data, (len(test_data), window, 1))
#     else:
#         test_data = np.reshape(test_data, (len(test_data), window))
#     return train_data, test_data


def createAE(encoding_dim=5, window=10):
    # Размерность кодированного представления
    # encoding_dim = 20

    # Энкодер
    # Входной плейсхолдер
    input_img = Input(window)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim)(flat_img)  # , activation='relu'

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    # flat_decoded = Dense(28 * 28, activation='sigmoid')(input_encoded)
    # decoded = Reshape((28, 28, 1))(flat_decoded)
    decoded = Dense(window)(input_encoded)  # , activation='sigmoid'

    # Модели, в конструктор первым аргументом передаются входные слои, а вторым выходные слои
    # Другие модели можно так же использовать как и слои
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def createDeepAE(encoding_dim=2, window=10):
    # Энкодер
    input_img = Input(shape=(window,))
    flat_img = Flatten()(input_img)
    x = None
    for i in range(window - 1, encoding_dim, -1):
        x = Dense(i)(flat_img)
    encoded = Dense(encoding_dim)(x)

    # Декодер
    input_encoded = Input(shape=(encoding_dim,))
    x = None
    for i in range(encoding_dim + 1, window):
        x = Dense(i)(input_encoded)
    # flat_decoded = Dense(window, activation='sigmoid')(x)
    flat_decoded = Dense(window)(x)
    decoded = Reshape((window,))(flat_decoded)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def createDeepConvAE(encoding_dim=49, window=10):
    input_img = Input(shape=(window, 1))

    x = Conv1D(10, 3, padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    encoded = Conv1D(1, 5, padding='same')(x)

    # На этом моменте представление  (7, 7, 1) т.е. 49-размерное
    input_encoded = Input(shape=(5, 1))
    x = Conv1D(10, 5, padding='same')(input_encoded)
    x = UpSampling1D(5)(x)
    decoded = Conv1D(1, window, padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def createSparseAE(encoding_dim=16, window=10):
    # encoding_dim = 16
    lambda_l1 = 0.000001

    # Энкодер
    input_img = Input(shape=(window,))
    flat_img = Flatten()(input_img)
    x = Dense(window - 2)(flat_img)  # , activation='relu'
    # x = Dense(encoding_dim * 2, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(lambda_l1))(x)

    # Декодер
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(encoding_dim - 2)(input_encoded)  # , activation='relu'
    # x = Dense(encoding_dim * 3, activation='relu')(x)
    flat_decoded = Dense(window)(x)  # , activation='sigmoid'
    decoded = Reshape((window,))(flat_decoded)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def plotResult(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])

    for row in range(5):
        fig, ax = plt.subplots()
        x1 = np.arange(0, window)
        random1 = int(random.random() * n)
        y1 = args[0][random1]
        y2 = args[1][random1]
        ax.plot(x1, y1, "rx", x1, y2, "b+", linestyle='solid')
        ax.plot(x1, [0.4 for i in range(window)], "go",
                x1, [-0.4 for i in range(window)], "go", linestyle='solid')
        ax.set_title('random state: ' + str(row))
        plt.show()


def createDenoisingModel(autoencoder, batch_size):
    def add_noise(x):
        noise_factor = 0.25
        x = x + K.random_normal(x.get_shape(), 0.25, noise_factor)
        x = K.clip(x, 0., 1.)
        return x

    input_img = Input(batch_shape=(batch_size, 10))
    noised_img = Lambda(add_noise)(input_img)

    noiser = Model(input_img, noised_img, name="noiser")
    denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
    return noiser, denoiser_model


def experimental1(input_data):
    x_train, x_test = input_data
    encoder, decoder, autoencoder = createAE(8)
    # encoder, decoder, autoencoder = createDeepAE(7)
    # encoder, decoder, autoencoder = createSparseAE(6)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # binary_crossentropy

    autoencoder.fit(x_train, x_train,
                    epochs=2000,
                    batch_size=500,
                    shuffle=True)

    imgs = x_test
    encoded_imgs = encoder.predict(imgs)
    decoded_imgs = decoder.predict(encoded_imgs)
    plotResult(imgs, decoded_imgs)

    # codes = encoder.predict(x_test)
    # sns_plot = sns.pairplot(pd.DataFrame(codes))
    # sns_plot.savefig('pairplot.png')
    plt.show()
    return encoder, decoder, autoencoder


def experimental2(input_data):
    x_train, x_test = input_data
    encoder, decoder, autoencoder = createDeepAE(7)

    batch_size = 100
    noiser, denoiser_model = createDenoisingModel(autoencoder, batch_size)
    denoiser_model.compile(optimizer='adam', loss='mean_squared_error')
    denoiser_model.fit(x_train, x_train,
                       epochs=2000,
                       batch_size=batch_size,
                       shuffle=True)