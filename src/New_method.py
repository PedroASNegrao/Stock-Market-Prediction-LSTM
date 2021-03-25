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
from plotly.subplots import make_subplots
import plotly.offline

import dash
import dash_core_components as dcc
import dash_html_components as html


class Predict:

    def __init__(self):
        # data_name = "XOM"
        data_name = "PETR4_SA_1"
        look_back = 15
        epochs_num = 25
        # switch_key = [False, False]  # [train, test]
        switch_key = [False, True]  # [train, test]
        # switch_key = [True, False]  # [train, test]
        # switch_key = [True, True] #[train, test]

        inicio = time.time()

        # test/train just one time
        self.NewMethod(epochs_num, data_name, look_back, switch_key)

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

        # print(len(close_train))
        # print(len(close_test))

        # Using TimeseriesGenerator to get the time series
        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
        valid_data_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=1)
        test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

        # train_generator_array = np.array(train_generator)
        # test_generator_arraytest_generator_array = np.array(test_generator)
        # print(close_train)
        # print(train_generator)

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

            # model.fit_generator(train_generator, epochs=epochs_num, verbose=2)

            # Saving the Neural network
            history = model.fit(train_generator, epochs=epochs_num, validation_data=valid_data_generator, verbose=1)
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
            print(close_test)
            print(prediction)
            # rmse = "TEST"

            test_array = []

            for i in range(len(close_test)-look_back):
                test_array.append(close_test[i])

            print(len(test_array))
            print(len(prediction))

            rmse = mean_squared_error(test_array, prediction, squared=False)
            results ="Data: {}, Epochs: {}, Look Back: {}, rmse: {}".format(data_name, epochs_num, look_back, rmse)
            with open('./../Models/{}/results.txt'.format(data_name), 'a') as result_manager:
                result_manager.write(results+'\n')

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
                title="{} - epochs: {} Stock - RMSE: {} - L.Back: {}".format(data_name, epochs_num, rmse, look_back),
                xaxis={'title': "Date"},
                yaxis={'title': "Close"}
            )

            fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)


            # fig.show()

            print("Loaded figure 1")
            # --------------------------------LOSS------------------------------------
            history = json.load(open("./../Models/{}/json/history-{}-{}.json".format(data_name, data_name, epochs_num), 'r'))

            epochs_array = []
            for i in range(epochs_num):
                epochs_array.append(i)

            # fig2 = make_subplots(rows=1, cols=2)
            # fig2.add_trace(go.Scatter(x=epochs_array, y=history['loss']), row=1, col=1)
            # fig2.add_trace(go.Scatter(x=epochs_array, y=history['val_loss']), row=1, col=1)
            # fig2.add_trace(go.Scatter(x=epochs_array, y=history['accuracy']), row=1, col=2)
            # fig2.add_trace(go.Scatter(x=epochs_array, y=history['val_accuracy']), row=1, col=2)

            traceL = go.Scatter(
                x=epochs_array,
                y=history['loss'],
                mode='lines',
                name='Predicted Loss'
            )
            traceL2 = go.Scatter(
                x=epochs_array,
                y=history['val_loss'],
                mode='lines',
                name='Real Loss'
            )
            layoutL = go.Layout(
                title="{} - epochs: {} Stock - RMSE: {}".format(data_name, epochs_num, rmse),
                xaxis={'title': "Date"},
                yaxis={'title': "Close"}
            )
            fig2 = go.Figure(data=[traceL, traceL2], layout=layoutL)

            app = dash.Dash()
            app.layout = html.Div([
                dcc.Graph(figure=fig),
                dcc.Graph(figure=fig2)
            ])

            app.run_server(debug=True, use_reloader=False)

            #
            # fig2.show()
            print("Loaded figure 2")

start = Predict()

