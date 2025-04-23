import os
import cv2
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
from utils import extract_landmarks, base_distance_transform
from typing import Callable


def create_dataset(path: str, transform: Callable[[np.ndarray], np.ndarray] = None) -> None:
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    annotations = pd.read_csv(os.path.join(path, 'annotations.csv'), header=None, names=['index', 'label'])
    data = {'x': [], 'y': []}

    print('Creating landmark dataset...')

    for row in annotations.iterrows():
        idx, label = row
        img = cv2.cvtColor(cv2.imread(os.path.join(path, f'{idx}.jpg')), cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(img, holistic)

        if landmarks:
            if transform:
                landmarks = transform(landmarks)

            data['x'].append(landmarks)
            data['y'].append(label)

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('Done!')


create_dataset('data', transform=base_distance_transform)