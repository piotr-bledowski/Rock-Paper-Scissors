import cv2
import sys

if __name__ == "__main__":
    print("=== Rock Paper Scissors Lizard Spock Game ===")
    print("1. Start Game")
    print("2. Test Player Detection")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        # Run the full game with player detection
        import interface
    elif choice == "2":
        # Run only the player detection test
        from player_detection import test_detection
        test_detection()
    elif choice == "3":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)
