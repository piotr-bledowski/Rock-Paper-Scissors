import os
import cv2
import time
import mediapipe as mp

"""
These functions are either purely cosmetic or simple enough to move them here
so as to not to clutter data_collection.py too much.
"""


PATH = os.path.join('dvc_module', 'tmp')

def count_samples() -> int:
    train_dirs = os.listdir(os.path.join(PATH, 'train'))
    test_dirs = os.listdir(os.path.join(PATH, 'test'))
    total_dirs = len(train_dirs) + len(test_dirs)
    return total_dirs


def create_directory(name):
    if not os.path.exists(os.path.join(PATH, name)):
        os.makedirs(os.path.join(PATH, name))


def add_window_text(image: cv2.UMat, action: str):
    cv2.putText(image, f"Collecting data for: {action}. Press SPACE to start recording,", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image,
                f"RIGHT ARROW to record next action, LEFT ARROW to record previous action,",
                (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image,
                f"ESC to exit.",
                (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def draw_landmarks(cap, holistic) -> cv2.UMat:
    success, image = cap.read()
    if not success:
        print("Failed to capture image.")
        return None

    image = cv2.flip(image, 1)  # show mirrored view for the sake of convenience
    # the following lines are apparently necessary for landmark drawing to work
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks,
                                              mp.solutions.holistic.HAND_CONNECTIONS)
    return image


def countdown(cap, holistic):
    start = time.time()
    while time.time() < start + 3:
        seconds_left = int(start + 3 - time.time()) + 1
        image = draw_landmarks(cap, holistic)
        cv2.putText(image, f'Begin in {seconds_left}', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('Camera', image)

        # without cv2.waitKey the window freezes for some reason
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break