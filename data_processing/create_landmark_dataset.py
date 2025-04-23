import os
import numpy as np
import mediapipe as mp
from utils import extract_landmarks
from tqdm import tqdm


def create_dataset(path: str) -> None:
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

    