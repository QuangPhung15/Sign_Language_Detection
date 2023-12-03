from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

# Root path for extracted data
DATA_PATH = "MP_DATA"
log_dir = "Logs"

# Videos per data
no_sequences = 30

# Frames per video
sequence_len = 30

def pre_process_data(actions):
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_len):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    return X_train, X_test, y_train, y_test

def train_neural_net(actions):
    tb_callback = TensorBoard(log_dir=log_dir)
    X_train, y_train, y_train, y_test = pre_process_data(actions)
    actions = np.array(actions)

    try:
        model = load_model("actions.h5")
    except: 
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=2000, callbacks=tb_callback)

    model.save("actions.h5")
