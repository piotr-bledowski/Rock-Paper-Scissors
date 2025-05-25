import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self):
        # Load the YOLO model (will download the small model if not present)
        self.model = YOLO("yolov8n.pt")
        # Person class ID in COCO dataset is 0
        self.person_class_id = 0
        
    def detect_players(self, frame):
        """
        Detect players in the frame and divide the frame into two halves
        
        Args:
            frame: Input frame from the camera
            
        Returns:
            player_present: Boolean indicating if two players are detected
            player1_frame: Left half of the frame (Player 1)
            player2_frame: Right half of the frame (Player 2)
            visualization: Frame with detection visualization
        """
        # Create a copy for visualization
        visualization = frame.copy()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Get all person detections
        persons = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == self.person_class_id:
                    # Get person bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Only add high confidence detections
                    if confidence > 0.5:
                        persons.append({
                            'bbox': (x1, y1, x2, y2),
                            'center_x': (x1 + x2) // 2,
                            'confidence': confidence
                        })
                    
                    # Draw bounding box on visualization
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(visualization, f"Person: {confidence:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if two or more persons are detected
        player_present = len(persons) >= 2
        
        # Draw game status on visualization
        if player_present:
            cv2.putText(visualization, "Two players detected! Ready to play!", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Sort persons by their x-coordinate
            persons.sort(key=lambda p: p['center_x'])
            
            # Draw player labels
            for i, person in enumerate(persons[:2]):
                x1, y1, x2, y2 = person['bbox']
                cv2.putText(visualization, f"Player {i+1}", 
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(visualization, f"Need two players! Detected: {len(persons)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Split the frame vertically into two halves for the two players
        height, width = frame.shape[:2]
        player1_frame = frame[:, :width//2]  # Left half
        player2_frame = frame[:, width//2:]  # Right half
        
        # Draw a vertical line separating the two halves on visualization
        cv2.line(visualization, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        return player_present, player1_frame, player2_frame, visualization

def test_detection():
    """Simple test function to demonstrate player detection"""
    cap = cv2.VideoCapture(0)
    detector = PlayerDetector()
    
    print("Testing player detection. Press ESC to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        player_present, player1_frame, player2_frame, viz = detector.detect_players(frame)
        
        # Show only the visualization frame with all info
        cv2.imshow("Player Detection Test", viz)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detection() 