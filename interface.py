import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from data_processing.utils import extract_landmarks, base_distance_transform
from player_detection import PlayerDetector
from game_logic import RPSLSGame, GameState
from game_gui import GameWindow

# Load the trained model
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

# Color per gesture
gesture_colors = {
    'rock': (0, 255, 0),
    'paper': (255, 0, 0),
    'scissors': (0, 0, 255),
    'lizard': (128, 0, 128),
    'spock': (0, 255, 255),
    'Detecting...': (100, 100, 100)
}

# Real-time landmark extraction
def extract_landmarks_xyz(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

def create_scaled_landmarks(original_landmarks, player_idx):
    """Create scaled landmarks for drawing on player's half of the screen"""
    from mediapipe.framework.formats import landmark_pb2
    
    scaled_landmarks = landmark_pb2.NormalizedLandmarkList()
    
    for landmark in original_landmarks.landmark:
        new_landmark = scaled_landmarks.landmark.add()
        # Scale x coordinate: remove offset and double the scale
        new_landmark.x = (landmark.x - (0.5 * player_idx)) * 2.0
        new_landmark.y = landmark.y
        new_landmark.z = landmark.z
        # Clamp x to valid range [0, 1]
        new_landmark.x = max(0.0, min(1.0, new_landmark.x))
    
    return scaled_landmarks

# Mediapipe Holistic setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                smooth_landmarks=True,
                                enable_segmentation=False,
                                refine_face_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

def process_frame(frame, game):
    """Process a single frame and return the processed frame"""
    # Detect players
    players_present, player1_frame, player2_frame, visualization = player_detector.detect_players(frame)
    
    # Create a display frame
    display_frame = visualization.copy()
    height, width = display_frame.shape[:2]
    half_width = width // 2
    
    # Draw a white vertical line separating the two halves on display frame
    cv2.line(display_frame, (half_width, 0), (half_width, height), (255, 255, 255), 2)
    
    # Only process gestures if two players are detected and game is active
    if players_present and game.state != GameState.WAITING_FOR_PLAYERS:
        # Process the full frame once with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # Create copies of each half for display
        player1_display = player1_frame.copy()
        player2_display = player2_frame.copy()
        
        # Process left and right hands and assign to the correct player based on position
        hands_data = []
        
        # Check left hand
        if results.left_hand_landmarks:
            # Determine which half the hand is in
            wrist_x = results.left_hand_landmarks.landmark[0].x
            player_idx = 0 if wrist_x < 0.5 else 1
            player_display = player1_display if player_idx == 0 else player2_display
            
            # Create scaled landmarks for the player's half
            scaled_landmarks = create_scaled_landmarks(results.left_hand_landmarks, player_idx)
            
            # Draw landmarks on the correct player's display with scaled coordinates
            mp_drawing.draw_landmarks(
                player_display, 
                scaled_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Process hand for gesture recognition (use original landmarks)
            landmarks = extract_landmarks_xyz(results.left_hand_landmarks)
            landmarks = np.array(base_distance_transform(landmarks))
            landmarks = (landmarks - np.min(landmarks)) / (np.max(landmarks) - np.min(landmarks) + 1e-6)
            
            # Predict gesture
            gesture = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks])) * 100
            
            # Only accept prediction if confidence > 50%
            if confidence > 50:
                final_gesture = gesture
            else:
                final_gesture = "Detecting..."
                
            hands_data.append({
                'player': player_idx + 1,
                'gesture': final_gesture,
                'confidence': confidence,
                'display': player_display
            })
        
        # Check right hand
        if results.right_hand_landmarks:
            # Determine which half the hand is in
            wrist_x = results.right_hand_landmarks.landmark[0].x
            player_idx = 0 if wrist_x < 0.5 else 1
            player_display = player1_display if player_idx == 0 else player2_display
            
            # Create scaled landmarks for the player's half
            scaled_landmarks = create_scaled_landmarks(results.right_hand_landmarks, player_idx)
            
            # Draw landmarks on the correct player's display with scaled coordinates
            mp_drawing.draw_landmarks(
                player_display, 
                scaled_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Process hand for gesture recognition (use original landmarks)
            landmarks = extract_landmarks_xyz(results.right_hand_landmarks)
            landmarks = np.array(base_distance_transform(landmarks))
            landmarks = (landmarks - np.min(landmarks)) / (np.max(landmarks) - np.min(landmarks) + 1e-6)
            
            # Predict gesture
            gesture = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks])) * 100
            
            # Only accept prediction if confidence > 50%
            if confidence > 50:
                final_gesture = gesture
            else:
                final_gesture = "Detecting..."
                
            hands_data.append({
                'player': player_idx + 1,
                'gesture': final_gesture,
                'confidence': confidence,
                'display': player_display
            })
        
        # Process the results for each player
        player_gestures = {}
        for player_idx in range(2):
            # Find hands for this player
            player_hands = [h for h in hands_data if h['player'] == player_idx + 1]
            player_display = player1_display if player_idx == 0 else player2_display
            
            if player_hands:
                # Use the hand with highest confidence
                best_hand = max(player_hands, key=lambda h: h['confidence'])
                player_gestures[player_idx] = best_hand['gesture']
                
                # Draw gesture on player's display frame
                color = gesture_colors.get(best_hand['gesture'], (255, 255, 255))
                cv2.putText(player_display, f"Player {player_idx+1}: {best_hand['gesture']} ({best_hand['confidence']:.0f}%)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # No hands detected for this player
                cv2.putText(player_display, f"Player {player_idx+1}: No hand detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # Copy the processed player frames into the display frame
        display_frame[:, :half_width] = player1_display
        display_frame[:, half_width:] = player2_display
        
        # Process round result if both players have gestures and round is active
        if len(player_gestures) == 2 and game.state == GameState.ROUND_ACTIVE:
            p1_gesture = player_gestures.get(0, "Detecting...")
            p2_gesture = player_gestures.get(1, "Detecting...")
            
            if p1_gesture != "Detecting..." and p2_gesture != "Detecting...":
                game.process_round_result(p1_gesture, p2_gesture)
    else:
        # If no players detected, show waiting message
        cv2.putText(display_frame, "Waiting for two players...", 
                   (width // 2 - 150, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return display_frame

# Initialize player detector
player_detector = PlayerDetector()

def main():
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
