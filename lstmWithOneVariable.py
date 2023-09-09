import matplotlib
import streamlit as st
import numpy as np
import requests as requests
import tensorflow as tf
from tensorflow import keras
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import matplotlib.dates as mdates
import random
from sklearn.metrics import r2_score

def run_lstm_model(data, predictOn):
    # Preprocess the data
    df = data[[predictOn]]
    df.loc[:, f'{predictOn}_norm'] = df[predictOn] / df[predictOn].max()
    sequence_length = 30
    train_size = int(len(df) * 0.85)
    train_df = df[:train_size]
    test_df = df[train_size - sequence_length:]

    # Define the function to create the sequences for input to the LSTM model
    def create_sequences(data, sequence_length):
        X = []
        y = []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

    # Create the training and testing sequences
    train_X, train_y = create_sequences(train_df[f'{predictOn}_norm'].values, sequence_length)
    test_X, test_y = create_sequences(test_df[f'{predictOn}_norm'].values, sequence_length)

    # Reshape the input data to have an additional dimension corresponding to the input dimension
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Define the function to create the LSTM model
    def create_lstm_model(units, dropout_rate):
        model = keras.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=True, input_shape=(sequence_length, 1)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        model.summary()
        return model

    # Define the number of iterations and parameter combinations to sample
    n_iter = 5

    # Define lists of potential values for each hyperparameter
    units_list = [16,32]
    dropout_rate_list = [0.2, 0.4]
    batch_size_list = [8, 16]

    # Initialize variables to store the best hyperparameters and best score
    best_hyperparameters = {}
    best_score = float('inf')  # Initialize with a large value for minimization tasks

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',   # Metric to monitor for early stopping (validation loss in this case)
        patience=5,            # Number of epochs to wait for improvement before stopping
        restore_best_weights=True   # Restore weights from the best epoch
    )

    # Perform randomized search cross-validation
    for _ in range(n_iter):
        # Randomly sample hyperparameter values from the defined lists
        units = random.choice(units_list)
        dropout_rate = random.choice(dropout_rate_list)
        epochs = 50
        batch_size = random.choice(batch_size_list)

        # Create the LSTM model with the sampled hyperparameters
        model = create_lstm_model(units=units, dropout_rate=dropout_rate)

        # Train the model with the sampled hyperparameters
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), callbacks=[early_stopping])

        # Use the trained model to make predictions on the test data
        test_pred = model.predict(test_X)

        # Evaluate the model's performance
        mse = tf.keras.metrics.mean_squared_error(test_y, test_pred).numpy()
        rmse = np.sqrt(mse)

        # Check if this set of hyperparameters is the best so far
        if (rmse < best_score).all():
            best_score = rmse
            best_hyperparameters = {
                'units': units,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }

    # Use the best hyperparameters to create the final LSTM model
    lstm_model = create_lstm_model(units=best_hyperparameters['units'], dropout_rate=best_hyperparameters['dropout_rate'])

    # Train the model with the best hyperparameters
    lstm_model.fit(train_X, train_y, epochs=best_hyperparameters['epochs'], batch_size=best_hyperparameters['batch_size'], validation_data=(test_X, test_y), callbacks=[early_stopping])

    # Use the trained model to make predictions on the test data
    test_pred = lstm_model.predict(test_X)

    # Evaluate the model's performance
    mse = tf.keras.metrics.mean_squared_error(test_y, test_pred).numpy()
    rmse = np.sqrt(mse)
    print('Root Mean Squared Error:', rmse)

    r2 = r2_score(test_y, test_pred)
    accuracy_percentage = r2 * 100

    st.write('Accuracy Percentage:', round(accuracy_percentage, 2), '%')

    fig, ax = plt.subplots()
    dates = test_df.index[sequence_length:]
    ax.plot(dates, test_y, label='Actual')
    ax.plot(dates, test_pred.reshape(-1), label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    return fig

