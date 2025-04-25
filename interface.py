import cv2
import mediapipe as mp
import numpy as np
# from joblib import load

# model = load('model.pkl')

# RPSLS logic
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

# Color per gesture
gesture_colors = {
    'rock': (0, 255, 0),
    'paper': (255, 0, 0),
    'scissors': (0, 0, 255),
    'lizard': (128, 0, 128),
    'spock': (0, 255, 255),
    'Detecting...': (100, 100, 100)
}

# Real-time landmark utilities (safe for mediapipe.Hands)
def extract_landmarks(hand_landmarks):
    return [[lm.x, lm.y] for lm in hand_landmarks.landmark]

def base_distance_transform(landmarks):
    base = np.array([landmarks[0][0], landmarks[0][1]])
    return np.array([np.linalg.norm(np.array([lm[0], lm[1]]) - base) for lm in landmarks])

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam start
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started – Press ESC to exit")
frame_count = 0  # Used for flashing animation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    height, width, _ = frame.shape
    hands_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = extract_landmarks(hand_landmarks)
            center_x = landmark_list[0][0]
            distances = base_distance_transform(landmark_list)

            # Placeholder for prediction
            # gesture = model.predict([distances])[0]
            # confidence = np.max(model.predict_proba([distances])) * 100
            gesture = "Detecting..."
            confidence = 0

            hands_data.append({'x': center_x, 'gesture': gesture, 'conf': confidence})

    if len(hands_data) == 2:
        hands_data.sort(key=lambda h: h['x'])
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

    cv2.imshow("RPSLS - Inference", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
