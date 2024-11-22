import os
import glob
from math import isinf
from hashlib import md5

import cv2
import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset


def valid(num: float) -> bool:
    return (num == num) and (not isinf(num))


# claude wrote this code, it's correct
def num_chunks(n: int, chunk_size: int, overlap_size: int) -> int:
    assert chunk_size > 0
    assert overlap_size >= 0
    assert n > 0

    assert n >= chunk_size
    assert chunk_size > overlap_size

    step_size = chunk_size - overlap_size
    num_chunks = ((n - (chunk_size - 1) - 1) // step_size) + 1

    assert num_chunks > 0
    return num_chunks


# TODO: move to pytest or something
assert (num_chunks(3, 2, 0)) == 1
assert (num_chunks(3, 2, 1)) == 2
assert (num_chunks(10, 5, 0)) == 2
assert (num_chunks(32, 8, 4)) == 7


def load_video(path: str):
    assert os.path.exists(path)

    if not os.path.exists("cache"):
        os.mkdir("cache")

    cache_name = md5(os.path.abspath(path).encode("utf-8")).hexdigest()
    cache_path = f"cache/{cache_name}"
    if os.path.exists(cache_path):
        return cache_path
    else:
        os.mkdir(cache_path)

    cap = cv2.VideoCapture(path)

    bar = tqdm.tqdm(total=20 * 60)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # frame = frame.astype("float32")
        # frame /= 255.0

        # frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (120, 90))
        cv2.imwrite(f"{cache_path}/{i}.jpg", frame)

        i += 1
        bar.update(1)

    bar.close()
    return cache_path


def load_cached(path: str):
    img = cv2.imread(path)
    img = img.astype("float32")
    img /= 255

    return img


def load_clip(path: str, start_frame: int, n_frames: int):
    assert n_frames > 0
    assert start_frame >= 0

    assert os.path.exists(path)
    frames_dir = load_video(path)

    frames = [
        f"{frames_dir}/{i}.jpg" for i in range(start_frame, start_frame + n_frames)
    ]
    frames = list(map(load_cached, frames))

    # for frame in frames:
    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(1)

    assert len(frames) == n_frames
    return frames


class CommaDataset(Dataset):
    def __init__(
        self, dir: str, chunk_size: int = 8, overlap_size: int = 0, transform=None
    ):
        assert chunk_size > 0
        assert overlap_size >= 0
        assert chunk_size > overlap_size

        self.dir = dir
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        self.transform = transform

        label_files = glob.glob(f"{dir}/*.txt")
        contiguous_chunks = []

        label_files = sorted(label_files)

        for fn in label_files:
            clipno = int(os.path.basename(fn).split(".")[0])
            f = open(fn, "r")

            points = f.readlines()
            points: list[tuple[float, float]] = [
                tuple(map(float, line.split())) for line in points
            ]
            npoints: list[tuple[int, int, tuple[float, float]]] = []

            # tag each line with clip number and frame number
            for frameno, pt in enumerate(points):
                # points[frameno] = (clipno, frameno, pt)
                npoints.append((clipno, frameno, pt))

            points: list[tuple[int, int, tuple[float, float]]] = npoints

            clip_sections = []
            section = []

            for pt in points:
                _, _, (pitch, yaw) = pt
                if not valid(pitch) or not valid(yaw):
                    if len(section) > self.chunk_size:
                        clip_sections.append(section)
                    section = []
                    continue

                section.append(pt)

            if len(section) > self.chunk_size:
                clip_sections.append(section)

            contiguous_chunks.extend(clip_sections)

        self.sections = contiguous_chunks
        self.section_chunk_counts = [
            num_chunks(len(chunk), self.chunk_size, self.overlap_size)
            for chunk in contiguous_chunks
        ]

        # sanity check that chunks are contiguous and have the same clip number
        for section in self.sections:
            clipno = section[0][0]
            for i in range(1, len(section)):
                assert section[i][0] == clipno
                assert section[i][1] == section[i - 1][1] + 1

    def __len__(self):
        return sum(self.section_chunk_counts)

    def __getitem__(self, i):
        section_idx = 0

        while i >= self.section_chunk_counts[section_idx]:
            i -= self.section_chunk_counts[section_idx]
            section_idx += 1

        section = self.sections[section_idx]
        step_size = self.chunk_size - self.overlap_size

        offset = i * step_size
        labels = section[offset : offset + self.chunk_size]

        clipno, start_frame, _ = labels[0]
        _, end_frame, _ = labels[-1]

        n_frames = end_frame - start_frame + 1

        assert n_frames == self.chunk_size
        labels = [label[2] for label in labels]
        labels = torch.tensor(labels)

        frames = load_clip(f"{self.dir}/{clipno}.hevc", start_frame, n_frames)
        out_frames = []

        if self.transform:
            for frame in frames:
                out_frames.append(self.transform(frame))

        frames = torch.stack(out_frames)
        return frames, labels


class DummyCommaDataset(Dataset):
    def __init__(self, length: int, chunk_size: int = 8):
        assert length > 0
        assert length >= chunk_size
        assert chunk_size > 0

        self.n = length
        self.chunk_size = chunk_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros((self.chunk_size, 3, 120, 90)), torch.zeros(
            (self.chunk_size, 2)
        )


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((120, 90)),
        ]
    )

    dataset = CommaDataset("./labeled", transform=transform)
    sample = dataset[0]
    print(sample[0].shape)
    print(sample[1].shape)
