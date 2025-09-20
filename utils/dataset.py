import glob
import os
import random
from os.path import split

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jedi.api.helpers import infer
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token

from utils.conversation import get_default_conv_template
from utils.attn_dataset import Attn_Dataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN)


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    label_path_list = []
    images_list = []
    images_clip_list = []
    salmap_list = []
    question_list = []
    answer_list = []
    conversation_list = []
    inferences = []
    for (
        image_path,
        label_path,
        image,
        image_clip,
        label,
        question,
        answer,
        conversation,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        label_path_list.append(label_path)
        images_list.append(image)
        images_clip_list.append(image_clip)
        salmap_list.append(label)
        conversation_list.extend(conversation)
        answer_list.append(answer)
        question_list.append(question)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(    # (2,471)
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    targets_what = input_ids.clone()
    targets_why = input_ids.clone()

    if conv_type == "llava_v1": # '###Assistant:'
        sep = conv.sep + conv.roles[1] + ": "   # '###Assistant: '
    else:
        sep = "[/INST] "
    for conversation, target, target_what, target_why in zip(conversation_list, targets, targets_what, targets_why):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        target_what[:cur_len] = IGNORE_INDEX
        target_why[:cur_len] = IGNORE_INDEX

        parts = conversation.split(sep)
        # if len(parts) != 2:
        #     break
        # assert len(parts) == 2, (len(parts), conversation)
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(conversation, tokenizer))     # 447
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2   # 391
        else:
            round_len = len(tokenizer(conversation).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        # Split answer into what and why components
        answer = parts[1]
        what_end = answer.find("3. Reason")
        if what_end == -1:
            what_end = answer.find("- Reason")
        if what_end == -1:
            what_end = answer.find("Reason")
        if what_end == -1:
            what_end = answer.find("3.")
        what_part = answer[:what_end] if what_end != -1 else answer
        why_part = answer[what_end:] if what_end != -1 else ""

        # Calculate token lengths
        if DEFAULT_IMAGE_TOKEN in conversation:
            what_len = len(tokenizer_image_token(what_part, tokenizer)) - 1
            why_len = len(tokenizer_image_token(why_part, tokenizer)) - 1
        else:
            what_len = len(tokenizer(what_part).input_ids) - 1
            why_len = len(tokenizer(why_part).input_ids) - 1


        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

        # Mask targets_what
        target_what[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        target_what[cur_len + instruction_len + what_len:] = IGNORE_INDEX

        # Mask targets_why
        target_why[cur_len: cur_len + instruction_len + what_len] = IGNORE_INDEX
        target_why[cur_len + instruction_len + what_len + why_len:] = IGNORE_INDEX

        cur_len += round_len - 1
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        # "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),    # (2,3,224,224)
        "gt_salmap": torch.stack(salmap_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "labels_what": targets_what,
        "labels_why": targets_why,
        "attention_masks": attention_masks,
        "questions_list": question_list,
        "answers_list": answer_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,     # path to W3DA
        tokenizer,
        vision_tower,
        samples_per_epoch=150 * 8 * 8 * 10,
        map_size: int = 256,
        precision: str = "fp32",
        image_size: int = 224,
        split: str = "training",
        dataset="BDDA||DReyeVE||LBW||DADA",
        sample_rate=[9, 3, 3, 1],
        eval_only=False,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.eval_only = eval_only

        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.split = split
        self.all_datasets = []
        self.dataset_lens = []
        for dataset in self.datasets:
            self.all_datasets.append(
                Attn_Dataset(
                    base_image_dir,
                    dataset,
                    tokenizer,
                    vision_tower,
                    samples_per_epoch,
                    map_size,
                    precision,
                    image_size,
                    split,
                    eval_only,
                )
            )
            self.dataset_lens.append(len(self.all_datasets[-1]))

    def __len__(self):
        if self.eval_only:
            data_size = sum(self.dataset_lens)
            return data_size
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.eval_only:
            cumulative = 0
            inference = True
            for i, length in enumerate(self.dataset_lens):
                if idx < cumulative + length:
                    dataset_idx = idx - cumulative
                    return *self.all_datasets[i][dataset_idx], inference
                cumulative += length
        else:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            data = self.all_datasets[ind]
            inference = False if self.split == "training" else True
            return *data[0], inference


