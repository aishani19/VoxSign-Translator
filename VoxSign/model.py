import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Bidirectional, Dense,
                                     Dropout, Input, LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Set the path to the data directory
PATH = os.path.join('data')

# Load actions from labels.json
labels_path = 'labels.json'
if os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        payload = json.load(f)
        if isinstance(payload, dict):
            actions = np.array(payload.get("actions", []))
        else:
            actions = np.array(payload)
else:
    actions = np.array(sorted([d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, d))]))

seed = int(os.environ.get("VOXSIGN_SEED", "42"))
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

print(f"Training on {len(actions)} actions")

frames = 10
landmarks, labels = [], []

for idx, action in enumerate(actions):
    action_path = os.path.join(PATH, action)
    if not os.path.exists(action_path):
        print(f"Warning: Skipping {action}, path not found")
        continue
    
    # Find all sequence directories
    sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    valid_sequences_count = 0
    for seq in sequences:
        seq_path = os.path.join(action_path, seq)
        # Check if it has all 10 frames
        if len([f for f in os.listdir(seq_path) if f.endswith('.npy')]) >= frames:
            temp = []
            for frame in range(frames):
                npy = np.load(os.path.join(seq_path, f"{frame}.npy"))
                temp.append(npy)
            landmarks.append(temp)
            labels.append(idx)
            valid_sequences_count += 1
    print(f"  Loaded {valid_sequences_count} sequences for {action}")

X = np.array(landmarks, dtype=np.float32)
Y = to_categorical(labels).astype(int)

print(f"Total samples: {len(X)}")

# Sequence-level normalization helps reduce signer/camera variance.
mean = np.mean(X, axis=(1, 2), keepdims=True)
std = np.std(X, axis=(1, 2), keepdims=True) + 1e-6
X = (X - mean) / std

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Model
model = Sequential(
    [
        Input(shape=(10, 126)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.35),
        Bidirectional(LSTM(96, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(256, activation="relu"),
        Dropout(0.30),
        Dense(128, activation="relu"),
        Dense(len(actions), activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
)

epochs = int(os.environ.get("VOXSIGN_EPOCHS", "80"))
callbacks = [
    EarlyStopping(monitor="val_top3_acc", patience=18, mode="max", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint("my_model.keras", monitor="val_top3_acc", mode="max", save_best_only=True),
]

y_train_labels = np.argmax(Y_train, axis=1)
class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(actions)),
    y=y_train_labels,
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_values)}

model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights,
    batch_size=32,
)

model.save('my_model.keras')

# Eval
probabilities = model.predict(X_test)
predictions = np.argmax(probabilities, axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)
top3 = metrics.top_k_accuracy_score(test_labels, probabilities, k=3, labels=np.arange(len(actions)))
print(f"Test Accuracy (Top-1): {accuracy}")
print(f"Test Accuracy (Top-3): {top3}")
