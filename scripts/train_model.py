import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import pickle


def load_data():
    base_dir = os.getcwd()
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Reshape data for Conv1D: (samples, timesteps, features=1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_test, y_train, y_test


def build_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train.shape[1:])
    
    # Calculate class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight_dict  # Applying class weights here
    )
    
    # Save training history for visualization
    with open('training_history_weighted.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Evaluating on test data:")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    model.save('final_scream_model_weighted.h5')
    print("Model saved as final_scream_model_weighted.h5")


if __name__ == "__main__":
    train()
