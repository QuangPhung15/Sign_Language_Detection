import mediapipe as mp    # Import mediapipe to get holistic model

# Root path for extracted data
DATA_PATH = "MP_DATA"

# Videos per data
no_sequences = 30

# Frames per video
sequence_len = 30

# Key points with MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities