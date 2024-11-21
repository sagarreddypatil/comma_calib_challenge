import cv2

def load_labels(path: str):
    with open(path, "r") as f:
        frames = f.readlines()

    frames = [tuple(map(float, frame.split())) for frame in frames]
    return frames
