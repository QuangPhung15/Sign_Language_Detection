from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import config as cf

def pre_process_data(actions):
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in range(cf.no_sequences):
            window = []
            for frame_num in range(cf.sequence_len):
                res = np.load(os.path.join(cf.DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            
            sequence.append(window)
            labels.append(label_map[action])
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    return X_train, X_test, y_train, y_test