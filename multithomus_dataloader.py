import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import os
import os.path
from tqdm import tqdm
import random
from utils import *
import math


def read_text_class(file, idx):
    secondary_classes = []
    with open(file) as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            sen = [s.strip() for s in line.split(",")]
            s_class = int(sen[1][idx:]) - 1
            secondary_classes.append(s_class)

    return secondary_classes


def make_dataset(split_file, split, root, num_classes=157):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset = []
    with open(split_file, "r") as f:
        data = json.load(f)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]["subset"] == split:
            if len(data[vid]["actions"]) < 1:
                continue
            fts = np.load(os.path.join(root, vid + ".npy"))
            num_feat = fts.shape[0]
            label = np.zeros((num_feat, num_classes), np.float32)

            #
            hmap = np.zeros((num_feat, num_classes), np.float32)
            action_lengths = []
            center_loc = []
            num_action = 0

            fps = num_feat / data[vid]["duration"]
            for ann in data[vid]["actions"]:
                #
                if ann[2] < ann[1]:
                    continue
                mid_point = (ann[2] + ann[1]) / 2
                for fr in range(0, num_feat, 1):
                    if fr / fps > ann[1] and fr / fps < ann[2]:
                        label[fr, ann[0] - 1] = (
                            1  # will make the first class to be the last for datasets other than Multi-Thumos #
                        )

                    if (fr + 1) / fps > mid_point and fr / fps < mid_point:
                        center = fr + 1
                        class_ = ann[0] - 1
                        action_duration = int((ann[2] - ann[1]) * fps)
                        radius = int(action_duration / gamma)
                        generate_gaussian(
                            hmap[:, class_], center, radius, tau, ku
                        )
                        num_action = num_action + 1
                        center_loc.append([center, class_])
                        action_lengths.append([action_duration])

            dataset.append(
                (
                    vid,
                    label,
                    data[vid]["duration"],
                    [
                        hmap,
                        num_action,
                        np.asarray(center_loc),
                        np.asarray(action_lengths),
                    ],
                )
            )
        i += 1

    return dataset


class MultiThomus(data_utl.Dataset):

    def __init__(
        self,
        cfg,
        input_type,
        data_split,
        data_percentage,
        num_clips,
        skip,
        shuffle=True,
        add_background=False,
    ):
        self.root = cfg.rgb_root
        self.data = make_dataset(
            cfg.annotations_file, data_split, cfg.rgb_root, cfg.num_classes
        )
        self.num_classes = int(cfg.num_classes)
        self.annotations = json.load(open(cfg.annotations_file, "r"))
        self.num_clips = int(cfg.num_clips)
        self.split = data_split
        assert data_split in ["training", "testing"]
        assert input_type in ["rgb", "flow", "combined"]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        entry = self.data[index]
        feat = np.load(os.path.join(self.root, entry[0] + ".npy"))
        feat = feat.reshape((feat.shape[0], 1, 1, feat.shape[-1]))
        features = feat.astype(np.float32)

        labels = entry[1]

        hmap, num_action, center_loc, action_lengths = entry[3]
        num_clips = self.num_clips
        masks = torch.zeros((num_clips))
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
        hmap = torch.from_numpy(hmap)

        if self.split in ["training", "testing"]:
            if len(features) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(
                        range(0, len(features) - num_clips)
                    )
                if self.split == "training":
                    features = features[
                        random_index : random_index + num_clips : 1
                    ]
                    labels = labels[random_index : random_index + num_clips : 1]
                    hmap = hmap[random_index : random_index + num_clips : 1]
                else:

                    features = features[
                        0 : len(features) - (len(features) % self.num_clips)
                    ]

                    labels = labels[
                        0 : len(labels) - (len(labels) % self.num_clips)
                    ]
                    hmap = labels[0 : len(hmap) - (len(hmap) % self.num_clips)]

                    features = torch.stack(
                        [
                            features[i : i + self.num_clips]
                            for i in range(0, len(features), self.num_clips)
                        ]
                    )
                    # features_2 = torch.stack(
                    #     [features_2[i:i + self.num_clips] for i in range(0, len(features_2), self.num_clips)])
                    labels = torch.stack(
                        [
                            labels[i : i + self.num_clips]
                            for i in range(0, len(labels), self.num_clips)
                        ]
                    )
                    hmap = torch.stack(
                        [
                            hmap[i : i + self.num_clips]
                            for i in range(0, len(hmap), self.num_clips)
                        ]
                    )
                    masks = torch.ones(features.shape[0], num_clips)
            elif len(features) < num_clips and self.split == "testing":
                masks = torch.zeros(num_clips)
                f = torch.zeros(
                    (
                        num_clips,
                        features.shape[1],
                        features.shape[2],
                        features.shape[3],
                    )
                )
                l = torch.zeros((num_clips, labels.shape[1]))
                h = torch.zeros((num_clips, hmap.shape[1]))

                f[: features.shape[0]] = features
                masks[: features.shape[0]] = 1
                l[: features.shape[0]] = labels
                h[: features.shape[0], :] = hmap[: features.shape[0], :]

                features = f
                labels = l
                hmap = h

        return features, masks, labels, [entry[0], entry[2], num_action], hmap

    def __len__(self):
        return len(self.data)


class collate_fn_unisize:

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def multithomus_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        # print(len(batch))
        new_batch = []
        length = random.choice(range(32, max_len, 16))
        for b in batch:
            f = np.zeros(
                (max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]),
                np.float32,
            )
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[2].shape[1]), np.float32)
            h = np.zeros((max_len, b[4].shape[1]), np.float32)
            f[: b[0].shape[0]] = b[0]
            m[: b[0].shape[0]] = 1
            l[: b[0].shape[0]] = b[2]
            h[: b[0].shape[0], :] = b[4][: b[0].shape[0], :]

            new_batch.append(
                [
                    video_to_tensor(f),
                    torch.from_numpy(m),
                    torch.from_numpy(l),
                    b[3],
                    torch.from_numpy(h),
                ]
            )

        return default_collate(new_batch)
