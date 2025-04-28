import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from data_processing.utils import extract_landmarks, base_distance_transform

#Load the trained model
model = load('gesture_model.pkl')

# RPSLS game rules
rules = {
    'rock': ['scissors', 'lizard'],
    'paper': ['rock', 'spock'],
    'scissors': ['paper', 'lizard'],
    'lizard': ['spock', 'paper'],
    'spock': ['scissors', 'rock']
}

def decide_winner(gesture1, gesture2):
    if gesture1 == gesture2:
        return "Draw"
    elif gesture2 in rules[gesture1]:
        return "Player 1 Wins"
    else:
        return "Player 2 Wins"

#Color per gesture
gesture_colors = {
    'rock': (0, 255, 0),
    'paper': (255, 0, 0),
    'scissors': (0, 0, 255),
    'lizard': (128, 0, 128),
    'spock': (0, 255, 255),
    'Detecting...': (100, 100, 100)
}

#Real-time landmark extraction
def extract_landmarks_xyz(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

#Mediapipe Holistic setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                smooth_landmarks=True,
                                enable_segmentation=False,
                                refine_face_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

#Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started – Press ESC to exit")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    height, width, _ = frame.shape
    hands_data = []

    #Process Left and Right hands separately
    for hand_type, hand_landmarks in [('left', results.left_hand_landmarks), ('right', results.right_hand_landmarks)]:
        if hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            landmarks = extract_landmarks_xyz(hand_landmarks)
            landmarks = np.array(base_distance_transform(landmarks))

            # if landmarks.shape[0] > 60:
            #     landmarks = landmarks[:60]
            # elif landmarks.shape[0] < 60:
            #     landmarks = np.pad(landmarks, (0, 60 - landmarks.shape[0]))

            center_x = hand_landmarks.landmark[0].x

            #Predict gesture
            gesture = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks])) * 100

            #Print real-time prediction
            print(f"{hand_type} hand - Predicted: {gesture} | Confidence: {confidence:.2f}%")

            #Only accept prediction if confidence > 60%
            if confidence > 60:
                final_gesture = gesture
            else:
                final_gesture = "Detecting..."

            hands_data.append({'x': center_x, 'gesture': final_gesture, 'conf': confidence})

    #Sort and decide winner
    if len(hands_data) == 2:
        hands_data.sort(key=lambda h: h['x'])  # leftmost = player 1
        p1, p2 = hands_data[0], hands_data[1]

        g1, g2 = p1['gesture'], p2['gesture']
        color1 = gesture_colors.get(g1, (255, 255, 255))
        color2 = gesture_colors.get(g2, (255, 255, 255))

        if g1 != "Detecting..." and g2 != "Detecting...":
            result = decide_winner(g1, g2)
        else:
            result = "Waiting for gestures..."

        cv2.putText(frame, f"Player 1: {g1} ({p1['conf']:.0f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
        cv2.putText(frame, f"Player 2: {g2} ({p2['conf']:.0f}%)", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)

        if result != "Waiting for gestures..." and result != "Draw":
            if (frame_count // 10) % 2 == 0:
                cv2.putText(frame, f"🏆 {result} 🏆", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        else:
            cv2.putText(frame, f"Result: {result}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    elif len(hands_data) == 1:
        cv2.putText(frame, "Waiting for second player...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    else:
        cv2.putText(frame, "No hands detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    cv2.imshow("RPSLS - Inference (Holistic)", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
