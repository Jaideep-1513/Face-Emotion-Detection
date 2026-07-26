"""
==========================================
Face Emotion Detection - Model Training
==========================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import model
from model import model

print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

# Load dataset
X_train = np.load("processed_data/X_train.npy")
y_train = np.load("processed_data/y_train.npy")

X_test = np.load("processed_data/X_test.npy")
y_test = np.load("processed_data/y_test.npy")

print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images :", X_test.shape)
print("Testing Labels :", y_test.shape)

# Create folder for saving model
os.makedirs("saved_model", exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(
    "saved_model/emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

print("\nTraining Completed!")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
