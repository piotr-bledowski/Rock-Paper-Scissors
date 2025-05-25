# Rock Paper Scissors Lizard Spock Game

A computer vision-based game that detects two players and their hand gestures to play Rock Paper Scissors Lizard Spock.

## Features

- Player detection using YOLOv8
- Automatic screen division for two players
- Hand gesture recognition using MediaPipe and machine learning
- Real-time game play with visual feedback

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the game:
   ```
   python main.py
   ```

3. Select an option:
   - 1: Start the full game with player detection
   - 2: Test only the player detection
   - 3: Exit

## How to Play

1. Two players need to be visible on the camera
2. Once detected, each player makes hand gestures on their side of the screen
3. The game will identify the gestures and determine the winner based on the rules:
   - Rock crushes Scissors and Lizard
   - Paper covers Rock and disproves Spock
   - Scissors cuts Paper and decapitates Lizard
   - Lizard poisons Spock and eats Paper
   - Spock smashes Scissors and vaporizes Rock

## Requirements

- Python 3.8+
- Webcam
- Sufficient lighting for player and hand detection

## Notes

- The YOLOv8 model will be downloaded automatically on first run (~15MB)
- For best results, ensure both players are clearly visible with good lighting
