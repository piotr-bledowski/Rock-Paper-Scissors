import numpy as np
import mediapipe as mp
import cv2


def extract_landmarks(img: np.ndarray, holistic: mp.solutions.holistic.Holistic) -> np.ndarray:
    results = holistic.process(img)
    r = results.right_hand_landmarks
    l = results.left_hand_landmarks

    landmarks = None

    if r:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in r.landmark]
    elif l:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in l.landmark]
    
    return np.array(landmarks)


def base_distance_transform(landmarks: np.ndarray) -> np.ndarray:
    result = []
    base = landmarks[0]

    for i in range(1, len(landmarks)):
        # calculate euclidean distance between each landmark and base
        result.append(np.linalg.norm(base, landmarks[i]))
    
    return result


img = cv2.imread('data/136.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(extract_landmarks(img).shape)
