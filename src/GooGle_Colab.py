import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import keras
import tensorflow as tf
import json
import time
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense
from keras import optimizers
import matplotlib.pyplot as plt

print("Inicio")


class Predict:

    def __init__(self):
        # data_name = "XOM"
        data_name = "PETR4_SA_1"
        look_back = 15
        epochs_num = 50
        batch_size =32
        neurons = 20
        data_path = "/content/Stock-Market-Prediction-LSTM/Data"
        result_path = "/content/drive/MyDrive/Neural_Network"

        # #Para rodar localmente:
        # data_path = "./../Data"
        # result_path = "./../Models"

        inicio = time.time()

        # test/train just one time
        self.NewMethod(epochs_num, data_name, look_back, result_path, data_path, batch_size, neurons)

        # # test/train an array
        # epochs_arrays = [50, 100, 200, 300, 400, 500]
        # look_back_array = [5, 10, 15, 20, 25]
        # batch_size_array = [1, 2, 4, 8, 16]

        # for ep in epochs_arrays:
        #     self.NewMethod(ep, data_name, look_back, result_path, data_path, batch_size, neurons)

        fim = time.time()
        tempo_total = (fim - inicio) / 60
        print("Tempo de execução foi de: %d minutos" % tempo_total)

    def NewMethod(self, epochs_num, data_name, look_back, result_path, data_path, batch_size, neurons):
        # --------------------------------TRAINING PART--------------------------------
        print("inicio do treio")
        df = pd.read_csv('{}/{}.csv'.format(data_path, data_name))
        # print(df.info())

        # Getting only Date and Close columns
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_axis(df['Date'], inplace=True)
        df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

        """
        #Plot close value
        df['Close'].plot(figsize=(16, 6), label='GOOG Close')
        plt.title("Test")
        plt.xlabel('Time')
        plt.ylabel('GOOG Stock Price')
        plt.legend()
        plt.show()
        """

        # --------------------------------DATA PROCESSING--------------------------------

        # Train the model on the first 80% of data and test it on the remaining 20%.

        close_data = df['Close'].values
        close_data = close_data.reshape((-1, 1))

        split_percent = 0.80
        split = int(split_percent * len(close_data))

        close_train = close_data[:split]
        close_test = close_data[split:]

        date_train = df['Date'][:split]
        date_test = df['Date'][split:]

        # print(len(close_train))
        # print(len(close_test))

        # Using TimeseriesGenerator to get the time series
        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=batch_size)
        valid_data_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=batch_size)
        test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=batch_size)

        # train_generator_array = np.array(train_generator)
        # test_generator_arraytest_generator_array = np.array(test_generator)
        # print(close_train)
        # print(train_generator)

        # --------------------------------NEURAL NETWORK--------------------------------
        model = Sequential()
        model.add(
            LSTM(neurons,
                 # activation = 'SineReLU',
                 activation='relu',
                 input_shape=(look_back, 1))
        )
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # model.fit_generator(train_generator, epochs=epochs_num, verbose=2)

        # Saving the Neural network
        history = model.fit(train_generator, epochs=epochs_num, validation_data=valid_data_generator, verbose=1)
        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict,
                  open("{}/{}/json/history-{}-epochs-{}-loockback-{}-batch_size-{}-neurons-{}.json".format(result_path, data_name, data_name,
                                                                                  epochs_num, look_back, batch_size, neurons), 'w'))
        # Save model
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/{}/json/regressor-{}-epochs-{}-loockback-{}-batch_size-{}-neurons-{}.json".format(result_path, data_name, data_name,
                                                                               epochs_num, look_back, batch_size, neurons),
                  "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(
            "{}/{}/h5/regressor-{}-epochs-{}-loockback-{}-batch_size-{}-neurons-{}.h5".format(result_path, data_name, data_name, epochs_num,
                                                                     look_back, batch_size, neurons))
        print("Saved model to disk")

        print("inicio do teste")
        # Load model
        # --------------------------------PREDICTION--------------------------------

        prediction = model.predict(test_generator)
        close_train = close_train.reshape((-1))
        close_test = close_test.reshape((-1))
        prediction = prediction.reshape((-1))

        test_array = []

        for i in range(len(close_test) - look_back):
            test_array.append(close_test[i])

        loss_rmse = mean_squared_error(history_dict['loss'], history_dict['val_loss'], squared=False)
        rmse = mean_squared_error(test_array, prediction, squared=False)

        # ---------------------------Criate-Graph------------------------------
        plt.plot(prediction)
        plt.plot(close_test)
        # plt.legend(['prediction', 'real'], loc='upper left')
        plt.legend(['prediction', 'real'])
        plt.ylabel('Stock-prices')
        plt.title("Data: {}, Epochs: {}, Look Back: {}, Batch Size: {}, Neurons: {}, rmse: {}".format(data_name, epochs_num, look_back, batch_size, neurons, rmse))
        plt.show()

        # summarize history for loss
        plt.plot(history_dict['loss'])
        plt.plot(history_dict['val_loss'])
        plt.title("Model loss: {}".format(loss_rmse))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # Results.txt

        results = "Data: {}, Epochs: {}, Look Back: {}, Batch Size: {}, Neurons: {}, rmse: {}, Loss_rmse: {}".format(data_name, epochs_num,
                                                                                        look_back, batch_size, neurons, rmse, loss_rmse)
        with open('{}/{}/results.txt'.format(result_path, data_name), 'a') as result_manager:
            result_manager.write(results + '\n')


start = Predict()

