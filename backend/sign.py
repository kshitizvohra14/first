import mediapipe as mp
import cv2
import numpy as np
import os
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create dataset directory
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hi', 'i love you', 'thank you'])  # Remove extra space
no_sequences = 30
sequence_length = 30

# Create folders
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)
for action in actions:
    for sequence in range(no_sequences):
        print(f"\nCollecting data for '{action}', sequence {sequence}")
        time.sleep(2)  # Delay before collecting

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract keypoints
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])

                    # Save keypoints
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)
                    print(f"Saved: {npy_path}")

            else:
                print(f"No hand detected for frame {frame_num} in sequence {sequence}.")

            # Add feedback to frame
            cv2.putText(image, f'{action} - Seq {sequence}, Frame {frame_num}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

