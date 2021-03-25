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

print("Inicio")

class Predict:

    def __init__(self):
        # data_name = "XOM"
        data_name = "PETR4_SA_1"
        look_back = 15
        epochs_num = 1

        inicio = time.time()

        # test/train just one time
        self.NewMethod(epochs_num, data_name, look_back)

        #test/train an array
        # epochs_arrays = [20, 30, 50, 100, 200, 300, 400, 500]
        # for ep in epochs_arrays:
        #     if ep == 100:
        #         switch_key = [False, True]
        #     else:
        #         switch_key = [True, True]
        #     self.NewMethod(ep, data_name, look_back, switch_key)


        fim = time.time()
        tempo_total = (fim - inicio) / 60
        print("Tempo de execução foi de: %d minutos" % tempo_total)

    def NewMethod(self, epochs_num, data_name, look_back):
        # --------------------------------TRAINING PART--------------------------------
        print("inicio do treio")
        df = pd.read_csv('/content/Stock-Market-Prediction-LSTM/Data/{}.csv'.format(data_name))
        #print(df.info())

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

        #print(len(close_train))
        #print(len(close_test))

        #Using TimeseriesGenerator to get the time series
        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
        valid_data_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=1)
        test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

        # train_generator_array = np.array(train_generator)
        # test_generator_arraytest_generator_array = np.array(test_generator)
        # print(close_train)
        # print(train_generator)

        # --------------------------------NEURAL NETWORK--------------------------------
        model = Sequential()
        model.add(
            LSTM(10,
                  # activation = 'SineReLU',
                  activation='relu',
                  input_shape=(look_back, 1))
        )
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        #model.fit_generator(train_generator, epochs=epochs_num, verbose=2)

        #Saving the Neural network
        history = model.fit(train_generator, epochs=epochs_num, validation_data=valid_data_generator, verbose=1)
        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict,
                  open("/content/drive/MyDrive/Neural_Network/Models/{}/json/history-{}-epochs-{}-loockback-{}.json".format(data_name, data_name, epochs_num, look_back), 'w'))
        # Save model
        # serialize model to JSON
        model_json = model.to_json()
        with open("/content/drive/MyDrive/Neural_Network/Models/{}/json/regressor-{}-epochs-{}-loockback-{}.json".format(data_name, data_name, epochs_num, look_back),
                  "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("/content/drive/MyDrive/Neural_Network/Models/{}/h5/regressor-{}-epochs-{}-loockback-{}.h5".format(data_name, data_name, epochs_num, look_back))
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

        print(len(test_array))
        print(len(prediction))

        #Results.txt
        rmse = mean_squared_error(test_array, prediction, squared=False)
        results ="Data: {}, Epochs: {}, Look Back: {}, rmse: {}".format(data_name, epochs_num, look_back, rmse)
        with open('/content/Stock-Market-Prediction-LSTM/Models/{}/results.txt'.format(data_name), 'a') as result_manager:
            result_manager.write(results+'\n')

start = Predict()
