# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List
import random
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from ._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotations(annotations_jsonpath, image_path):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    file = open(annotations_jsonpath)
    
    # Build an index which maps image id with a list of caption annotations.
    entries = []
    for line in file:
        lineLS = eval(line)
        if os.path.exists(image_path + lineLS[0] + '.npy'):
            entries.append(
                {
                    "caption": lineLS[1].lower(),
                    "foil": lineLS[-1],
                    "image_id": lineLS[0],
                }
            )
    return entries

def get_image_features(reader):
    item = {}
    item["image_id"] = reader.item().get("image_id")
    item["image_h"] = reader.item().get("image_height")
    item["image_w"] = reader.item().get("image_width")
    item["num_boxes"] = reader.item().get("num_boxes")
    item["boxes"] = reader.item().get("bbox")
    item["features"] = reader.item().get("features")
    image_h = int(item["image_h"])
    image_w = int(item["image_w"])
    # num_boxes = int(item['num_boxes'])

    # features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
    # boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
    features = item["features"].reshape(-1, 2048)
    boxes = item["boxes"].reshape(-1, 4)

    num_boxes = features.shape[0]
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1
    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location_ori = copy.deepcopy(image_location)
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
    image_location_ori = np.concatenate(
        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
    )

    return features, num_boxes, image_location, image_location_ori

class MultimodalTwitterDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=20,
        max_region_num=101,
        clean_datasets=None
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.image_path = dataroot
        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self._max_region_num = max_region_num
        self.num_labels = 2
        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + ".pkl",
            )

        if not os.path.exists(cache_path):
            self._entries = _load_annotations(annotations_jsonpath, dataroot)
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))
        print('load data', len(self._entries))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            # sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            # sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in sentence_tokens
            # ]
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]
        reader = np.load(self.image_path + image_id + '.npy', allow_pickle=True)
        features, num_boxes, boxes, _ = get_image_features(reader)
        # self._image_features_reader[image_id]

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        caption = entry["token"]
        target = int(entry["foil"])
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            int(image_id),
        )

    def __len__(self):
        return len(self._entries)
