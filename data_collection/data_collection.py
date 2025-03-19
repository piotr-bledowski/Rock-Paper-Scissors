import os
import numpy as np
import mediapipe as mp
from pynput import keyboard

from helpers import *
from actions import ACTION_TO_IDX


class DataCollector:
    def __init__(self, tmp_path):
        self.space_pressed = False
        self.right_arrow_pressed = False
        self.left_arrow_pressed = False
        self.tmp_path = tmp_path
        self.actions = np.array(list(ACTION_TO_IDX.keys()))
        self.action_idx = 0


    def on_press(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = True
        elif key == keyboard.Key.right:
            self.right_arrow_pressed = True
        elif key == keyboard.Key.left:
            self.left_arrow_pressed = True


    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = False
        if key == keyboard.Key.right:
            self.right_arrow_pressed = False
        elif key == keyboard.Key.left:
            self.left_arrow_pressed = False


    def handle_action_change(self):
        if self.right_arrow_pressed:
            if self.action_idx < len(self.actions) - 1:
                self.action_idx += 1
            else:
                self.action_idx = 0
            cv2.waitKey(200)

        if self.left_arrow_pressed:
            if self.action_idx > 0:
                self.action_idx -= 1
            else:
                self.action_idx = len(self.actions) - 1
            cv2.waitKey(200)


    def annotate_sample(self, sample_num: int, action: str):
        with open(os.path.join(self.tmp_path, f'annotations.csv'), 'a') as f:
            f.write(f'{sample_num},{action}\n')


    def save_image(self, cap, sample_num, frame):
        _, image = cap.read()
        if frame is not None:
            image = cv2.resize(image, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.tmp_path, str(sample_num), f'{frame}.jpg'), image)


    def record_samples(self):
        samples = [int(dir) for dir in os.listdir(self.tmp_path) if os.path.isdir(os.path.join(self.tmp_path, dir))]
        n_samples = 0 if not samples else sorted(samples)[len(samples)-1]+1
        sequences = 500  # max number of samples to record
        frames = 100  # max number of frames per sample
        cap = cv2.VideoCapture(0)

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            action = self.actions[self.action_idx]
            listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()

            for sequence in range(sequences):
                while not self.space_pressed:
                    image = draw_landmarks(cap, holistic)
                    add_window_text(image, action)
                    cv2.imshow('Camera', image)

                    # Select action
                    self.handle_action_change()
                    action = self.actions[self.action_idx]

                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                os.makedirs(f'{os.path.join(self.tmp_path, str(n_samples))}')
                print(f"Collection data for: {action} sequence no: {sequence}.")
                countdown(cap, holistic)

                # Collect sample
                for frame in range(frames):
                    self.save_image(cap, n_samples, frame)
                    image = draw_landmarks(cap, holistic)
                    cv2.putText(image, f"Collecting frames. Action: {action} frame no: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f"Press ESC to stop recording", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Camera', image)
                    if cv2.waitKey(1) == 113:  # q to stop recording
                        break
                self.annotate_sample(n_samples, action)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    print('Finished recording. To push to remote from the default directory, run py dvc.py --command push')
                    break

                n_samples += 1

        cap.release()
        cv2.destroyAllWindows()
        listener.stop()
