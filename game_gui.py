import sys
import cv2
import numpy as np
import mediapipe as mp
from joblib import load
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QRadioButton, 
                            QButtonGroup, QFrame, QGridLayout, QSizePolicy, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from game_logic import RPSLSGame, GameState
from player_detection import PlayerDetector
from data_processing.utils import extract_landmarks, base_distance_transform

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.running = True
        self.player_detector = PlayerDetector()
        
        # Load the trained model
        self.model = load('gesture_model.pkl')
        
        # Color per gesture
        self.gesture_colors = {
            'rock': (0, 255, 0),
            'paper': (255, 0, 0),
            'scissors': (0, 0, 255),
            'lizard': (128, 0, 128),
            'spock': (0, 255, 255),
            'Detecting...': (100, 100, 100)
        }
        
        # Mediapipe Holistic setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks_xyz(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    def create_scaled_landmarks(self, original_landmarks, player_idx):
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
        
    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Convert frame to RGB
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
        cap.release()
        
    def process_frame(self, frame):
        """Process a single frame and return the processed frame"""
        # Mirror the frame horizontally for more intuitive display
        frame = cv2.flip(frame, 1)
        
        # Detect players
        players_present, player1_frame, player2_frame, visualization = self.player_detector.detect_players(frame)
        
        # Create a display frame from the original frame (not visualization)
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        half_width = width // 2
        
        # Always draw the white vertical line separating the two halves
        cv2.line(display_frame, (half_width, 0), (half_width, height), (255, 255, 255), 3)
        
        # Always show player detection boxes if available
        if players_present:
            # Draw bounding boxes on the display frame
            # Get the bounding boxes from the player detector
            try:
                # Extract bounding box information from the visualization
                # We'll overlay the detection boxes on our display frame
                detection_overlay = visualization.copy()
                
                # Create masks for the left and right halves
                left_mask = detection_overlay[:, :half_width]
                right_mask = detection_overlay[:, half_width:]
                
                # Apply the detection visualization to our display frame
                display_frame[:, :half_width] = cv2.addWeighted(
                    display_frame[:, :half_width], 0.7, left_mask, 0.3, 0
                )
                display_frame[:, half_width:] = cv2.addWeighted(
                    display_frame[:, half_width:], 0.7, right_mask, 0.3, 0
                )
            except:
                pass
            
            # Process gestures if game is active
            if self.game.state != GameState.WAITING_FOR_PLAYERS and self.game.is_recognition_active():
                # Process each half separately
                player_gestures = {}
                
                # Process Player 1 (left half)
                if player1_frame is not None:
                    left_half = player1_frame.copy()
                    gesture1, confidence1 = self.process_player_frame(left_half, 1)
                    if gesture1:
                        player_gestures[0] = gesture1
                        
                        # Draw gesture info on left half
                        color = self.gesture_colors.get(gesture1, (255, 255, 255))
                        cv2.putText(display_frame, f"Player 1: {gesture1} ({confidence1:.0f}%)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        cv2.putText(display_frame, "Player 1: No hand detected", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                
                # Process Player 2 (right half)
                if player2_frame is not None:
                    right_half = player2_frame.copy()
                    gesture2, confidence2 = self.process_player_frame(right_half, 2)
                    if gesture2:
                        player_gestures[1] = gesture2
                        
                        # Draw gesture info on right half
                        color = self.gesture_colors.get(gesture2, (255, 255, 255))
                        cv2.putText(display_frame, f"Player 2: {gesture2} ({confidence2:.0f}%)", 
                                   (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        cv2.putText(display_frame, "Player 2: No hand detected", 
                                   (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
                
                # Process round result if both players have gestures and round is active
                if len(player_gestures) == 2 and self.game.state == GameState.ROUND_ACTIVE:
                    p1_gesture = player_gestures.get(0, "Detecting...")
                    p2_gesture = player_gestures.get(1, "Detecting...")
                    
                    if p1_gesture != "Detecting..." and p2_gesture != "Detecting...":
                        self.game.process_round_result(p1_gesture, p2_gesture)
            elif self.game.state in [GameState.COUNTDOWN, GameState.ROUND_BREAK, GameState.ROUND_END]:
                # Show status during non-active periods
                if self.game.state == GameState.COUNTDOWN:
                    cv2.putText(display_frame, f"Get ready! Round starts in {self.game.countdown_value}...", 
                               (width // 2 - 200, height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                elif self.game.state == GameState.ROUND_BREAK:
                    cv2.putText(display_frame, "Round break - Get ready for next round!", 
                               (width // 2 - 250, height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                elif self.game.state == GameState.ROUND_END:
                    cv2.putText(display_frame, f"{self.game.round_result}", 
                               (width // 2 - 150, height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # Show waiting for game start message
                cv2.putText(display_frame, "Two players detected - Ready to start!", 
                           (width // 2 - 200, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # If no players detected, show waiting message
            cv2.putText(display_frame, "Waiting for two players...", 
                       (width // 2 - 150, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Always redraw the vertical line to ensure it's visible
        cv2.line(display_frame, (half_width, 0), (half_width, height), (255, 255, 255), 3)
        
        return display_frame
    
    def process_player_frame(self, player_frame, player_num):
        """Process a single player's frame and return gesture and confidence"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(player_frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            # Check for hand landmarks (prioritize right hand, then left hand)
            hand_landmarks = None
            if results.right_hand_landmarks:
                hand_landmarks = results.right_hand_landmarks
                # Draw landmarks on the player frame
                self.mp_drawing.draw_landmarks(
                    player_frame, 
                    hand_landmarks, 
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            elif results.left_hand_landmarks:
                hand_landmarks = results.left_hand_landmarks
                # Draw landmarks on the player frame
                self.mp_drawing.draw_landmarks(
                    player_frame, 
                    hand_landmarks, 
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            if hand_landmarks:
                # Extract and process landmarks for gesture recognition
                landmarks = self.extract_landmarks_xyz(hand_landmarks)
                landmarks = np.array(base_distance_transform(landmarks))
                landmarks = (landmarks - np.min(landmarks)) / (np.max(landmarks) - np.min(landmarks) + 1e-6)
                
                # Predict gesture
                gesture = self.model.predict([landmarks])[0]
                confidence = np.max(self.model.predict_proba([landmarks])) * 100
                
                # Only accept prediction if confidence > 50%
                if confidence > 50:
                    return gesture, confidence
                else:
                    return "Detecting...", confidence
            
            return None, 0
            
        except Exception as e:
            print(f"Error processing player {player_num} frame: {e}")
            return None, 0
        
    def stop(self):
        self.running = False
        self.wait()

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rock Paper Scissors Lizard Spock")
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a2e, stop:0.3 #16213e, stop:0.6 #0f3460, stop:1 #533483);
                border-image: url() 0 0 0 0 stretch stretch;
            }
            QLabel {
                color: #e8f4fd;
                font-size: 14px;
                font-weight: normal;
                text-shadow: 0px 0px 8px rgba(135, 206, 250, 0.6);
                padding: 4px;
                margin: 2px;
                background: transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e3c72, stop:0.5 #2a5298, stop:1 #1e3c72);
                color: #ffffff;
                border: 2px solid #4a90e2;
                padding: 10px 20px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
                text-shadow: 0px 0px 6px rgba(74, 144, 226, 0.8);
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a5298, stop:0.5 #4a90e2, stop:1 #2a5298);
                border: 2px solid #87ceeb;
                text-shadow: 0px 0px 10px rgba(135, 206, 235, 1.0);
                box-shadow: 0px 0px 15px rgba(74, 144, 226, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a365d, stop:0.5 #2c5282, stop:1 #1a365d);
                border: 2px solid #2980b9;
            }
            QPushButton:disabled {
                background: #2d3748;
                color: #718096;
                border: 2px solid #4a5568;
                text-shadow: none;
            }
            QRadioButton {
                color: #e8f4fd;
                font-size: 14px;
                font-weight: normal;
                text-shadow: 0px 0px 6px rgba(135, 206, 250, 0.6);
                spacing: 8px;
                padding: 4px;
                margin: 2px;
                background: transparent;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #4a90e2;
                border-radius: 10px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    stop:0 rgba(26, 54, 93, 0.3), stop:1 rgba(74, 144, 226, 0.1));
            }
            QRadioButton::indicator:checked {
                border: 2px solid #87ceeb;
                border-radius: 10px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    stop:0 #4a90e2, stop:0.7 #2a5298, stop:1 #1e3c72);
                box-shadow: 0px 0px 8px rgba(74, 144, 226, 0.8);
            }
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 60, 114, 0.4), stop:0.5 rgba(74, 144, 226, 0.2), stop:1 rgba(42, 82, 152, 0.3));
                border: 2px solid rgba(74, 144, 226, 0.5);
                border-radius: 15px;
                margin: 4px;
                padding: 12px;
                box-shadow: inset 0px 0px 20px rgba(135, 206, 250, 0.1);
            }
            #titleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #87ceeb;
                text-shadow: 0px 0px 15px rgba(135, 206, 235, 0.8);
                padding: 10px;
                margin: 10px;
                background: transparent;
            }
            #scoreLabel {
                font-size: 18px;
                font-weight: bold;
                color: #00d4ff;
                text-shadow: 0px 0px 12px rgba(0, 212, 255, 0.8);
                padding: 6px;
                margin: 4px;
                background: transparent;
            }
            #resultLabel {
                font-size: 16px;
                font-weight: bold;
                color: #40e0d0;
                text-shadow: 0px 0px 10px rgba(64, 224, 208, 0.8);
                padding: 8px;
                margin: 4px;
                min-height: 40px;
                background: transparent;
            }
            #stateLabel {
                font-size: 14px;
                font-weight: normal;
                color: #b0e0e6;
                text-shadow: 0px 0px 8px rgba(176, 224, 230, 0.6);
                padding: 6px;
                margin: 4px;
                min-height: 30px;
                background: transparent;
            }
            #roundLabel {
                font-size: 16px;
                font-weight: bold;
                color: #add8e6;
                text-shadow: 0px 0px 10px rgba(173, 216, 230, 0.8);
                padding: 6px;
                margin: 4px;
                background: transparent;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel for game controls and info
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Game title
        title_label = QLabel("🎮 Rock Paper Scissors\nLizard Spock 🎮")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        left_layout.addWidget(title_label)
        
        # Round selection
        rounds_frame = QFrame()
        rounds_layout = QVBoxLayout(rounds_frame)
        rounds_layout.setSpacing(10)
        rounds_label = QLabel("🎯 Select Number of Rounds:")
        rounds_label.setWordWrap(True)
        rounds_layout.addWidget(rounds_label)
        
        self.rounds_group = QButtonGroup()
        rounds_grid = QGridLayout()
        rounds_grid.setSpacing(10)
        rounds = [3, 5, 7, 9]
        for i, num in enumerate(rounds):
            radio = QRadioButton(f"{num} Rounds")
            if i == 0:  # Select first option by default
                radio.setChecked(True)
            self.rounds_group.addButton(radio, i)
            rounds_grid.addWidget(radio, i//2, i%2)
        rounds_layout.addLayout(rounds_grid)
        left_layout.addWidget(rounds_frame)
        
        # Countdown toggle
        countdown_frame = QFrame()
        countdown_layout = QVBoxLayout(countdown_frame)
        countdown_layout.setSpacing(5)
        
        self.countdown_checkbox = QCheckBox("⏰ Enable Round Countdown")
        self.countdown_checkbox.setChecked(True)  # Default enabled
        self.countdown_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e8f4fd;
                font-size: 14px;
                font-weight: normal;
                text-shadow: 0px 0px 6px rgba(135, 206, 250, 0.6);
                spacing: 8px;
                padding: 4px;
                margin: 2px;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #4a90e2;
                border-radius: 4px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    stop:0 rgba(26, 54, 93, 0.3), stop:1 rgba(74, 144, 226, 0.1));
            }
            QCheckBox::indicator:checked {
                border: 2px solid #87ceeb;
                border-radius: 4px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    stop:0 #4a90e2, stop:0.7 #2a5298, stop:1 #1e3c72);
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjUgNEw2IDExLjVMMi41IDgiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
            }
        """)
        countdown_layout.addWidget(self.countdown_checkbox)
        left_layout.addWidget(countdown_frame)
        
        # Start button
        self.start_button = QPushButton("🚀 Start Game")
        self.start_button.clicked.connect(self.start_game)
        left_layout.addWidget(self.start_button)
        
        # Game info
        self.game_info = QFrame()
        game_info_layout = QVBoxLayout(self.game_info)
        game_info_layout.setSpacing(10)
        
        # Round display
        self.round_label = QLabel("🏆 Round: -")
        self.round_label.setObjectName("roundLabel")
        self.round_label.setWordWrap(True)
        game_info_layout.addWidget(self.round_label)
        
        # Score display
        self.score_label = QLabel("📊 Score: 0 - 0")
        self.score_label.setObjectName("scoreLabel")
        self.score_label.setWordWrap(True)
        game_info_layout.addWidget(self.score_label)
        
        # Game state
        self.state_label = QLabel("⏳ Game State: Waiting")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setWordWrap(True)
        game_info_layout.addWidget(self.state_label)
        
        # Result display
        self.result_label = QLabel("")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        game_info_layout.addWidget(self.result_label)
        
        left_layout.addWidget(self.game_info)
        left_layout.addStretch()
        
        # Reset button
        self.reset_button = QPushButton("🔄 New Game")
        self.reset_button.clicked.connect(self.reset_game)
        self.reset_button.setEnabled(False)
        left_layout.addWidget(self.reset_button)
        
        # Right panel for video feed
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        
        # Video display
        self.video_label = QLabel("📹 Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("""
            QLabel {
                background: qradialgradient(cx:0.5, cy:0.5, radius:1.0,
                    stop:0 rgba(10, 10, 46, 0.8), stop:0.7 rgba(22, 33, 62, 0.6), stop:1 rgba(15, 52, 96, 0.4));
                border: 3px solid rgba(74, 144, 226, 0.7);
                border-radius: 15px;
                font-size: 16px;
                color: #87ceeb;
                padding: 20px;
                text-shadow: 0px 0px 12px rgba(135, 206, 235, 0.8);
            }
        """)
        right_layout.addWidget(self.video_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Initialize game
        self.game = None
        self.video_thread = None
        
        # Set window size
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        
    def start_game(self):
        if not self.rounds_group.checkedButton():
            return
            
        rounds = int(self.rounds_group.checkedButton().text().split()[0])
        self.game = RPSLSGame()
        
        # Set countdown preference
        self.game.set_countdown_enabled(self.countdown_checkbox.isChecked())
        
        self.game.start_game(rounds)
        
        # Start video thread
        self.video_thread = VideoThread(self.game)
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.start()
        
        # Disable start button and enable reset
        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        
        # Start game update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game_state)
        self.timer.start(100)  # Update every 100ms
        
    def reset_game(self):
        """Reset the game to initial state"""
        if self.video_thread:
            self.video_thread.stop()
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.game:
            self.game.stop_timers()
            
        self.game = None
        self.video_thread = None
        
        # Reset UI
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.round_label.setText("🏆 Round: -")
        self.score_label.setText("📊 Score: 0 - 0")
        self.state_label.setText("⏳ Game State: Waiting")
        self.result_label.setText("")
        self.video_label.setText("📹 Camera feed will appear here")
        
    def update_video(self, qt_image):
        # Scale image to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_game_state(self):
        if not self.game:
            return
            
        # Update round display
        self.round_label.setText(f"🏆 Round: {self.game.current_round}/{self.game.rounds_to_play}")
        
        # Update score display
        self.score_label.setText(f"📊 Score: {self.game.player1_score} - {self.game.player2_score}")
        
        # Update game state
        state_text = {
            GameState.WAITING_FOR_PLAYERS: "⏳ Waiting for Players",
            GameState.ROUND_SETUP: "🎯 Get Ready!",
            GameState.COUNTDOWN: f"⏰ Round {self.game.current_round} starting in {self.game.countdown_value}...",
            GameState.ROUND_ACTIVE: "🎮 Round Active - Make your gesture!",
            GameState.ROUND_END: f"✅ Round {self.game.current_round} Complete",
            GameState.ROUND_BREAK: "⏸️ Preparing for next round...",
            GameState.GAME_END: "🏁 Game Over"
        }.get(self.game.state, "Unknown")
        self.state_label.setText(f"Game State: {state_text}")
        
        # Update result display
        if self.game.state == GameState.GAME_END:
            winner_emoji = "🏆" if self.game.game_result != "Game Draw" else "🤝"
            self.result_label.setText(f"{winner_emoji} {self.game.game_result}\n🏆 Final Score: {self.game.player1_score} - {self.game.player2_score}")
        elif self.game.state == GameState.ROUND_END and self.game.round_result:
            result_emoji = "🎉" if self.game.round_result != "Draw" else "🤝"
            self.result_label.setText(f"{result_emoji} {self.game.round_result}")
        else:
            self.result_label.setText("")
            
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'game') and self.game:
            self.game.stop_timers()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 