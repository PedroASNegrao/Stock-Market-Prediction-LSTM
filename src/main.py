import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from keras.models import model_from_json
import json

# Import Keras modules
# Initializing the neural network:
from keras.models import Sequential
# Adding a densely connected neural network layer:
from keras.layers import Dense
# Adding the Long Short-Term Memory layer:
from keras.layers import LSTM
# Adding dropout layers that prevent overfitting:
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import mean_squared_error


class Predict:

    def __init__(self):
        data_name = "PETR4"
        epochs_num = 500

        inicio = time.time()
        #test/train an array
        """
        epochs_arrays = [10,100,300,200,500]
        for ep in epochs_arrays:
            self.train(ep)
            self.test(ep)
        """

        #test/train just one time
        #self.train(epochs_num, data_name)
        self.test(epochs_num, data_name)


        fim = time.time()
        tempo_total = (fim - inicio) / 60
        print("Tempo de execução foi de: %d minutos" % tempo_total)

    def train(self, epochs_num, data_name):
        # --------------------------------TRAINING PART--------------------------------
        print("inicio do treio")
        # Loading the Dataset
        dataset_train = pd.read_csv('./../Data/{}.csv'.format(data_name))
        training_set = dataset_train.iloc[:, 1:2].values

        # Scale the dataset to numbers between zero and one
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)

        # LSTMs expect our data to be in a specific format, usually a 3D array with timesteps

        # Create data in 60 timesteps
        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Convert the Date to a #D array
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        # print(X_train)

        regressor = Sequential()
        regressor.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
        regressor.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False))
        regressor.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        regressor.add(Dense(1, kernel_initializer='uniform', activation='linear'))

        optimizer = optimizers.Adam(clipvalue=0.5)
        regressor.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        # regressor.fit(X_train, y_train, epochs=epochs_num, batch_size=32)
        # history = regressor.fit(X_train, y_train, epochs=epochs_num, batch_size=32)

        history = regressor.fit(X_train, y_train, validation_split=0.33, epochs=epochs_num, batch_size=32, verbose=1)

        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict, open("./../Models/{}/json/history-{}-{}.json".format(data_name,data_name, epochs_num), 'w'))
        # Save model
        # serialize model to JSON
        model_json = regressor.to_json()
        with open("./../Models/{}/json/regressor-{}-{}.json".format(data_name, data_name, epochs_num), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        regressor.save_weights("./../Models/{}/h5/regressor-{}-{}.h5".format(data_name, data_name, epochs_num))
        print("Saved model to disk")

        # --------------------------------TESTING PART--------------------------------

    def test(self, epochs_num, data_name):

        print("Inicio do teste")
        # Load model
        json_file = open("./../Models/{}/json/regressor-{}-{}.json".format(data_name, data_name, epochs_num), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        regressor = model_from_json(loaded_model_json)
        # load weights into new model
        regressor.load_weights("./../Models/{}/h5/regressor-{}-{}.h5".format(data_name, data_name, epochs_num))
        print("Loaded model from disk")

        dataset_train = pd.read_csv('./../Data/{}.csv'.format(data_name))
        # Scale the dataset to numbers between zero and one
        sc = MinMaxScaler(feature_range=(0, 1))

        # Predicting Future Stock using the Test Set
        dataset_test = pd.read_csv('./../Data/{}-test.csv'.format(data_name))
        real_stock_price = dataset_test.iloc[:, 1:2].values

        #get the length of the csv
        data_array = dataset_test.to_numpy()
        test_lenght = len(data_array)

        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

        inputs = inputs.reshape(-1, 1)
        # inputs = sc.fit(inputs)
        obj = sc.fit(inputs)
        inputs = obj.transform(inputs)
        X_test = []

        for i in range(60, 60+test_lenght):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        #PREDICTION
        #----------------------------------------------------
        X_test_new =[]
        num_prediction = 30
        look_back = 60
        prediction_list = X_test[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            #x = x.reshape((1, look_back, 1))
            out = regressor.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back - 1:]

        last_date = dataset_test['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        #return prediction_dates

        #-------------------------------------------------------

        history = json.load(open("./../Models/{}/json/history-{}-{}.json".format(data_name, data_name, epochs_num), 'r'))

        #print(type(history))

        # summarize history for accuracy
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy epc:%d' % epochs_num)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss epc:%d' % epochs_num)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        rmse = mean_squared_error(real_stock_price, predicted_stock_price, squared=False)

        # Plotting the Results
        dataset_test['Open'].plot(figsize=(16, 6))
        #dataset_test['Close'].plot(figsize=(16, 6))
        #predicted_stock_price.plot(figsize=(16, 6))
        #plt.plot(real_stock_price, color='black', label='PETR4-2.0 Stock Price')
        plt.plot(predicted_stock_price, color='green', label='Predicted PETR4-2.0 Stock Price')
        plt.title(f'PETR4-2.0 Stock Price Prediction - epc:{epochs_num} (RMSE: {rmse})')
        plt.xlabel('Time')
        plt.ylabel('PETR4 Stock Price')
        plt.legend()
        plt.show()

        rmse = mean_squared_error(real_stock_price, predicted_stock_price, squared=False)
        print("rmse do modelo geral: %f" % rmse)


start = Predict()
