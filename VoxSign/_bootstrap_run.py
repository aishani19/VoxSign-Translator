"""One-off: synthetic data + quick train so main.py can run. Safe to delete after."""
import os
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

PATH = os.path.join("data")
actions = np.array(["a", "b"])
sequences, frames = 30, 10

for action, sequence, frame in product(actions, range(sequences), range(frames)):
    d = os.path.join(PATH, action, str(sequence))
    os.makedirs(d, exist_ok=True)
    base = np.ones(126, dtype=np.float32) * (1.0 if action == "a" else -1.0)
    noise = np.random.randn(126).astype(np.float32) * 0.08
    np.save(os.path.join(d, str(frame)), base + noise)

label_map = {label: num for num, label in enumerate(actions)}
landmarks, labels = [], []
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        temp.append(np.load(os.path.join(PATH, action, str(sequence), str(frame) + ".npy")))
    landmarks.append(temp)
    labels.append(label_map[action])

X = np.array(landmarks)
Y = to_categorical(labels).astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=34, stratify=Y
)

model = Sequential()
model.add(LSTM(32, return_sequences=True, activation="relu", input_shape=(10, 126)))
model.add(LSTM(64, return_sequences=True, activation="relu"))
model.add(LSTM(32, return_sequences=False, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.fit(X_train, Y_train, epochs=8, verbose=1)
model.save("my_model.keras")
print("Saved my_model.keras; you can run: python main.py")
