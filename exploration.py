import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from math import degrees

from dataset import load_labels


def labels():
    frames = load_labels("labeled/1.txt")
    pitches = [frame[0] for frame in frames]
    yaws = [frame[1] for frame in frames]

    pitches = list(map(degrees, pitches))
    yaws = list(map(degrees, yaws))

    # frames are tuples of (pitch, yaw), line plot
    # sns.lineplot(x=range(len(pitches)), y=pitches)
    sns.lineplot(x=range(len(yaws)), y=yaws)

    plt.show()


def video():
    cap = cv2.VideoCapture("labeled/1.hevc")
    labels = load_labels("labeled/1.txt")

    # make live chart
    plot = plt.figure()
    ax = plot.add_subplot(111)
    plt.ion()

    plt.show()

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (120, 90))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frame = cv2.resize(frame, (640, 480))

        labels_till_now = labels[:i + 1]
        yaws = [frame[1] for frame in labels_till_now]

        ax.clear()
        sns.lineplot(x=range(len(yaws)), y=yaws, ax=ax)
        plot.canvas.draw()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        i += 1


if __name__ == "__main__":
    # labels()
    video()
