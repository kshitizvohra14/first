import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ----- CONFIGURATION -----
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hi', 'i love you', 'thank you'])
no_sequences = 30
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}

# ----- LOAD DATA -----
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                res = np.load(path)
                if res.shape == (63,):
                    window.append(res)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ----- NORMALIZE INPUT -----
X = X / np.max(X)

# ----- TRAIN TEST SPLIT -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- BUILD MODEL -----
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----- CALLBACKS -----
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5)

# ----- TRAIN MODEL -----
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
          callbacks=[checkpoint, earlystop])

# ----- SAVE FINAL MODEL -----
model.save('sign_language_model.h5')
