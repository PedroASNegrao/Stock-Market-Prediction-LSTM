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
import plotly.graph_objects as go


class Predict:

    def __init__(self):
        data_name = "PETR4_SA_1"
        look_back = 30
        epochs_num = 1
        switch_key = [True, False] #[train, test]

        inicio = time.time()

        # test/train just one time
        self.NewMethod(epochs_num, data_name, look_back, switch_key)

        #test/train an array
        epochs_arrays = [1,50,100,200,300,500]
        for ep in epochs_arrays:
            self.NewMethod(ep, data_name, look_back, switch_key)




        fim = time.time()
        tempo_total = (fim - inicio) / 60
        print("Tempo de execução foi de: %d minutos" % tempo_total)

    def NewMethod(self, epochs_num, data_name, look_back, switch_key):
        # --------------------------------TRAINING PART--------------------------------
        print("inicio do treio")
        df = pd.read_csv('./../Data/{}.csv'.format(data_name))
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
        test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

        # --------------------------------NEURAL NETWORK--------------------------------
        if(switch_key[0] == True):
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
            history = model.fit(train_generator, epochs=epochs_num, verbose=1)
            # Get the dictionary containing each metric and the loss for each epoch
            history_dict = history.history
            # Save it under the form of a json file
            json.dump(history_dict,
                      open("./../Models/{}/json/history-{}-{}.json".format(data_name, data_name, epochs_num), 'w'))
            # Save model
            # serialize model to JSON
            model_json = model.to_json()
            with open("./../Models/{}/json/regressor-{}-{}.json".format(data_name, data_name, epochs_num),
                      "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("./../Models/{}/h5/regressor-{}-{}.h5".format(data_name, data_name, epochs_num))
            print("Saved model to disk")


        # --------------------------------TESTING PART--------------------------------
        if (switch_key[1] == True):
            print("inicio do teste")
            # Load model
            json_file = open("./../Models/{}/json/regressor-{}-{}.json".format(data_name, data_name, epochs_num), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("./../Models/{}/h5/regressor-{}-{}.h5".format(data_name, data_name, epochs_num))
            print("Loaded model from disk")
            # --------------------------------PREDICTION--------------------------------

            prediction = model.predict(test_generator)

            close_train = close_train.reshape((-1))
            close_test = close_test.reshape((-1))
            prediction = prediction.reshape((-1))


        # --------------------------------FORECASTING--------------------------------
            num_prediction = 30

            def predict(num_prediction, model):
                prediction_list = close_data[-look_back:]

                for _ in range(num_prediction):
                    x = prediction_list[-look_back:]
                    x = x.reshape((1, look_back, 1))
                    out = model.predict(x)[0][0]
                    prediction_list = np.append(prediction_list, out)
                prediction_list = prediction_list[look_back - 1:]

                return prediction_list

            def predict_dates(num_prediction):
                last_date = df['Date'].values[-1]
                prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
                return prediction_dates

            forecast = predict(num_prediction, model)
            forecast_dates = predict_dates(num_prediction)
            close_data = close_data.reshape((-1))

            # --------------------------------PLOTING------------------------------------
            print(len(close_test))
            print(len(prediction))
            rmse = "TEST"
            #rmse = mean_squared_error(close_train, prediction, squared=False)

            trace1 = go.Scatter(
                x=date_train,
                y=close_train,
                mode='lines',
                name='Data'
            )
            trace2 = go.Scatter(
                x=date_test,
                y=prediction,
                mode='lines',
                name='Prediction'
            )
            trace3 = go.Scatter(
                x=date_test,
                y=close_test,
                mode='lines',
                name='Ground Truth'
            )
            trace4 = go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode='lines',
                name='Forecast'
            )
            layout = go.Layout(
                title="{} - epochs: {} Stock - RMSE: {}".format(data_name, epochs_num, rmse),
                xaxis={'title': "Date"},
                yaxis={'title': "Close"}
            )
            fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
            fig.show()

            # --------------------------------LOSS------------------------------------
            #history = json.load(open("./../Models/{}/json/history-{}-{}.json".format(data_name, data_name, epochs_num), 'r'))

            # epochs_array = []
            # for i in range(epochs_num):
            #     epochs_array.append(i)
            #
            # trace_L1 = go.Scatter(
            #     x=epochs_array,
            #     y=history['loss'],
            #     mode='lines',
            #     name='Predicted Loss'
            # )
            # trace_L2 = go.Scatter(
            #     x=epochs_array,
            #     y=history['val_loss'],
            #     mode='lines',
            #     name='Real Loss'
            # )
            # layout_L = go.Layout(
            #     title="{} Loss".format(data_name),
            #     xaxis={'title': "epoch"},
            #     yaxis={'title': "loss"}
            # )
            # fig2 = go.Figure(data=[trace_L1, trace_L2], layout=layout_L)
            # fig2.show()

start = Predict()
