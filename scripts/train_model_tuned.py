import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D,
    LSTM, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
import pickle


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test  = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # Reshape for Conv1D: (samples, timesteps, features=1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    return X_train, X_test, y_train, y_test


def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu',
               kernel_regularizer=l2(1e-4),
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation='relu',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation='relu',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(2),

        # Increased LSTM capacity
        LSTM(96, return_sequences=False,
             kernel_regularizer=l2(1e-4)),
        Dropout(0.5),

        # Larger Dense layer
        Dense(128, activation='relu',
              kernel_regularizer=l2(1e-4)),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train.shape[1:])

    # Fixed moderate class weights (tune if needed)
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    print("Class counts (train): 0 =", neg, ", 1 =", pos)

    weight_for_0 = 1.0
    weight_for_1 = 2.0   # emphasize screams without over-bias
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print("Using class_weight:", class_weight)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model_tuned.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
    )

    with open('training_history_tuned.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Evaluating on test data:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")

    model.save('final_scream_model_tuned.h5')
    print("Model saved as final_scream_model_tuned.h5")


if __name__ == "__main__":
    train()
