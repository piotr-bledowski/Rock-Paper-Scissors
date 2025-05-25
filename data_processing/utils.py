import numpy as np
import mediapipe as mp
import cv2


def extract_landmarks(img: np.ndarray, holistic: mp.solutions.holistic.Holistic) -> np.ndarray:
    results = holistic.process(img)
    r = results.right_hand_landmarks
    l = results.left_hand_landmarks

    if r:
        return [[landmark.x, landmark.y, landmark.z] for landmark in r.landmark]
    if l:
        return [[landmark.x, landmark.y, landmark.z] for landmark in l.landmark]
    return None


def base_distance_transform(landmarks: np.ndarray) -> np.ndarray:
    result = []
    base = landmarks[0]
    middle_finger_tip = landmarks[12]
    ring_finger_tip = landmarks[16]

    for i in range(1, len(landmarks)):
        # calculate euclidean distance between each landmark and base
        result.append(np.sqrt(np.sum((np.array(base) - np.array(landmarks[i])) ** 2)))
    
    result.append(np.sqrt(np.sum((np.array(middle_finger_tip) - np.array(ring_finger_tip)) ** 2)))
    
    return result
