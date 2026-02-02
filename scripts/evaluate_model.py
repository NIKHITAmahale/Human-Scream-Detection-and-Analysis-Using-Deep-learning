import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import os


def plot_training(history):
    """Plot training vs validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()


def load_test_data():
    """Load test data from data/processed folder relative to project root."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # For Conv1D: (samples, timesteps, 1)
    X_test = X_test[..., np.newaxis]
    return X_test, y_test


def evaluate_at_threshold(model, X_test, y_test, threshold=0.45):
    """Evaluate model at a given decision threshold and print metrics."""
    # Predict probabilities
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    # Convert to class labels using threshold
    y_pred = (y_pred_prob > threshold).astype(int)

    print("\n" + "=" * 60)
    print(f"Final evaluation at threshold = {threshold:.2f}")
    print("=" * 60)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)


def evaluate_model():
    """Main evaluation pipeline: load model, history, data, and evaluate."""
    # 1. Load test data
    X_test, y_test = load_test_data()

    # 2. Load trained model
    model = load_model('final_scream_model_tuned.h5')

    # 3. Load training history
    with open('training_history_tuned.pkl', 'rb') as f:
        history = pickle.load(f)

    # 4. Plot training curves
    plot_training(history)

    # 5. Evaluate once at the chosen best threshold (0.45)
    evaluate_at_threshold(model, X_test, y_test, threshold=0.45)


if __name__ == "__main__":
    evaluate_model()