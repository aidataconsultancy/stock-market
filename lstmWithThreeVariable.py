import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objs as go
import random
from sklearn.metrics import r2_score

def run_lstm_model_multiVariable(data):
    # Ejecutando predicción con modelo LSTM con 3 variables: Close, Open y Volume
    # Preprocesamiento de los datos
    df3 = data[['Close', 'Open', 'Volume']]
    df3_norm = df3 / df3.max()
    sequence_length3 = 30
    train_size3 = int(len(df3_norm) * 0.85)
    train_df3 = df3_norm[:train_size3]
    test_df3 = df3_norm[train_size3 - sequence_length3:]

   # Define the function to create the sequences for input to the LSTM model
    def create_sequences(data3, sequence_length3):
        X3 = []
        y3 = []
        for i in range(len(data3) - sequence_length3):
            x_seq3 = data3[i:i + sequence_length3, :]
            X3.append(x_seq3)
            y3.append(data3[i + sequence_length3, 0])
        return np.array(X3), np.array(y3)

    # Create the training and testing sequences
    train_data3 = train_df3.values
    test_data3 = test_df3.values
    train_X3, train_y3 = create_sequences(train_data3, sequence_length3)
    test_X3, test_y3 = create_sequences(test_data3, sequence_length3)

    # Build the LSTM model
    num_features3 = train_X3.shape[2]

    # Define the hyperparameters for randomized search
    param_grid = {
        'units': [32, 64],
        'dropout_rate': [0.2, 0.4],
        'batch_size': [16, 32]
    }

    best_model = None
    best_rmse3 = float('inf')

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',   # Metric to monitor for early stopping (validation loss in this case)
        patience=5,            # Number of epochs to wait for improvement before stopping
        restore_best_weights=True   # Restore weights from the best epoch
    )

    # Perform randomized search cross-validation
    n_iter = 5  # Number of hyperparameter combinations to try
    for _ in range(n_iter):
        # Randomly select hyperparameters
        epochs = 50
        batch_size = random.choice(param_grid['batch_size'])
        units = random.choice(param_grid['units'])
        dropout_rate = random.choice(param_grid['dropout_rate'])

        model3 = keras.Sequential([
            tf.keras.layers.LSTM(units if num_features3 == 3 else units // 2, return_sequences=True,
                        input_shape=(sequence_length3, num_features3)),
            tf.keras.layers.LSTM(units // 2, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units // 4, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units // 8),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model3.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        model3.summary()

        # Train the model
        model3.fit(train_X3, train_y3, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(test_X3, test_y3), callbacks=[early_stopping])

        # Use the trained model to make predictions on the test data
        test_pred3 = model3.predict(test_X3)

        # Evaluate the model's performance
        mse3 = tf.keras.metrics.mean_squared_error(test_y3, test_pred3).numpy()
        rmse3 = np.sqrt(mse3)

        # Update best model if current model performs better
        if (rmse3 < best_rmse3).all():
            best_rmse3 = rmse3
            best_model = model3

    # Train the best model with the full training set
    best_model.fit(train_X3, train_y3, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(test_X3, test_y3), callbacks=[early_stopping])

    # Use the trained model to make predictions on the test data
    test_pred3 = best_model.predict(test_X3)

    # Evaluate the model's performance
    mse3 = tf.keras.metrics.mean_squared_error(test_y3, test_pred3).numpy()
    rmse3 = np.sqrt(mse3)
    print('Root Mean Squared Error:', rmse3)

    r2 = r2_score(test_y3, test_pred3)
    accuracy_percentage = r2 * 100

    st.write('Porcentaje de Exactitud con Múltiples Variables:', round(accuracy_percentage, 2), '%')

    # Creando gráfico interactivo con plotly
    dates = test_df3.index[sequence_length3:]
    trace_actual = go.Scatter(x=dates, y=test_y3, mode='lines', name='Real')
    trace_pred = go.Scatter(x=dates, y=test_pred3.reshape(-1), mode='lines', name='Predicho')
    layout = go.Layout(title='Comparación Real vs. Predicho', xaxis=dict(title='Fecha'), yaxis=dict(title='Precio'))
    fig3 = go.Figure(data=[trace_actual, trace_pred], layout=layout)

    st.plotly_chart(fig3)  # Muestra el gráfico en Streamlit

    return fig3

