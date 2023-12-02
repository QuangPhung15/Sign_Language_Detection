# Import all dependencies
import cv2                                  # Import OpenCV to work with camera
import numpy as np                          # Import numpy for ...
import os                                   # Import OS to work with directory easier
import time                                 # Import time for machine learning frame
from matplotlib import pyplot as pyplot     # Import matplotlib to visual image
import config as cf

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color conversion
    image.flags.writeable = False  # Save memory
    results = model.process(image) # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color reversion

    return image, results

def draw_landmarks(image, results):
    cf.mp_drawing.draw_landmarks(image, results.face_landmarks, cf.mp_holistic.FACEMESH_CONTOURS) # Draw face landmarks
    cf.mp_drawing.draw_landmarks(image, results.pose_landmarks, cf.mp_holistic.POSE_CONNECTIONS) # Draw pose landmarks
    cf.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, cf.mp_holistic.HAND_CONNECTIONS) # Draw left hand landmarks
    cf.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, cf.mp_holistic.HAND_CONNECTIONS) # Draw right hand landmarks

def draw_styled_landmarks(image, results):
    # Draw face landmarks
    cf.mp_drawing.draw_landmarks(image, results.face_landmarks, cf.mp_holistic.FACEMESH_CONTOURS, 
                              cf.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              cf.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    # Draw pose landmarks
    cf.mp_drawing.draw_landmarks(image, results.pose_landmarks, cf.mp_holistic.POSE_CONNECTIONS, 
                              cf.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              cf.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)) 
    
    # Draw left hand landmarks
    cf.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, cf.mp_holistic.HAND_CONNECTIONS, 
                              cf.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              cf.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)) 
    
    # Draw right hand landmarks
    cf.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, cf.mp_holistic.HAND_CONNECTIONS, 
                              cf.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              cf.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)) 

def extract_keypoints(results):
    face, pose, right_hand, left_hand = np.zeros(33 * 4), np.zeros(468 * 3), np.zeros(21 * 3), np.zeros(21 * 3)

    # Extract pose landmarks
    if (results.pose_landmarks):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    
    # Extract face landmarks
    if (results.face_landmarks):
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()

    # Extract left hand landmarks
    if (results.left_hand_landmarks):
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    
    # Extract right hand landmarks
    if (results.right_hand_landmarks):
        left_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose, face, left_hand, right_hand])

def setupFolder(actions):
    for action in actions:
        for i in range(cf.no_sequences):
            try:
                os.makedirs(os.path.join(cf.DATA_PATH, action, str(i)))
            except:
                pass

def collectKeypoints(actions):
    # Create camera object
    cap = cv2.VideoCapture(0)

    # Create data folders
    setupFolder(actions)

    # Set mediapipe holistic model
    with cf.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # LOOP through actions
        for action in actions:
            # LOOP through each sequences (training videos)
            for sequence in range(cf.no_sequences):
                # LOOP through video length (number of frames = 30)
                for frame_num in range(cf.sequence_len):

            
                    # Read frame
                    ret, frame = cap.read()

                    # Make detection
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks (face, pose, left and right hands)
                    draw_styled_landmarks(image, results)

                    if (frame_num == 0):
                        cv2.putText(image, "START NEW COLLECTION", (120, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f"Collecting frames for {action} Video Number {sequence}", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), cv2.LINE_AA)
                    else:
                        cv2.putText(image, f"Collecting frames for {action} Video Number {sequence}", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), cv2.LINE_AA)

                    # Show on the screen
                    cv2.imshow("Sign Language Detection", image)
                    
                    # Saving key points
                    keyPoints = extract_keypoints(results)
                    np_path = os.path.join(cf.DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(np_path, keyPoints)

                    # Terminate
                    if (cv2.waitKey(10) & 0xFF == ord("q")):
                        break
        
        cap.release()
        cv2.destroyAllWindows()