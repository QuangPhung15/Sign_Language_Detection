# Import all dependencies
import cv2                                  # Import OpenCV to work with camera
import numpy as np                          # Import numpy for ...
import os                                   # Import OS to work with directory easier
import mediapipe as mp                      # Import mediapipe to get holistic model
import time                                 # Import time for machine learning frame
from matplotlib import pyplot as pyplot     # Import matplotlib to visual image

# Key points with MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_holistic.FACEMESH_CONTOURS
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion
    image.flags.writeable = False  # Save memory
    results = model.process(image) # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color reversion

    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand landmarks

def main():
    cap = cv2.VideoCapture(0)
    # Set mediapipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            
            # Read frame
            ret, frame = cap.read()

            # Make detection
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            draw_landmarks(image, results)

            # Show on the screen
            cv2.imshow("Sign Language Detection", image)

            # Terminate
            if (cv2.waitKey(10) & 0xFF == ord("q")):
                break
        
        cap.release()
        cv2.destroyAllWindows()

main()