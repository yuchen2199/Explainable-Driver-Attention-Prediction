import os
import random
from torchvision import transforms
import numpy as np
import pickle
import json
import cv2
import pandas as pd
import torch
from transformers import CLIPImageProcessor
from utils.utils import (ATTN_TOKEN)

from model.llava import conversation as conversation_lib
from utils.utils import BLANK_QUESTION, DEFAULT_IMAGE_TOKEN, ANSWER_LIST
from tqdm import tqdm

RAW_FRAME_TYPE = 'raw_frames'
LABEL_FRAME_TYPE = 'gazemap_frames'
QUESTION_DIR = 'questions'
ANSWER_DIR = 'answers'

class Attn_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,     # path to W3DA
        dataset,            # BDDA/DReyeVE/LBW/DADA
        tokenizer,
        vision_tower,
        samples_per_epoch=150 * 8 * 8 * 10,
        map_size=256,
        precision: str = "fp32",
        image_size: int = 224,
        split: str = "training",
        eval_only=False,    # For evaluation
    ):
        self.samples_per_epoch = samples_per_epoch
        self.eval_only = eval_only

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.split = split
        self.dataset = dataset
        self.answer_list = ANSWER_LIST

        self.map_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((map_size, map_size)),
            transforms.ToTensor()
        ])

        dataset_dir = os.path.join(base_image_dir, dataset)
        self.split_dir = os.path.join(dataset_dir, split)
        self.dataset_dir = dataset_dir
        print(f'Loading samples of dataset {dataset}')
        self.images, self.labels, self.questions, self.answers = self.init_attn()

    def __len__(self):
        if self.eval_only:
            return len(self.labels)
        return self.samples_per_epoch

    def init_attn(self):
        rf_path_list, lf_path_list, question_list, answer_list = self._load_from_source()

        return rf_path_list, lf_path_list, question_list, answer_list

    def _load_from_source(self):
        rf_path_list = []
        lf_path_list = []
        question_list = []
        answer_list = []

        split_dir = self.split_dir
        vids = sorted(os.listdir(split_dir))

        for vid in tqdm(vids, desc=f"Loading data from videos Directory", ncols=80):
            vid_dir = os.path.join(split_dir, vid)
            if not os.path.exists(os.path.join(vid_dir, RAW_FRAME_TYPE)):
                continue
            json_list = sorted(os.listdir(os.path.join(vid_dir, RAW_FRAME_TYPE)))

            for json_file in json_list:
                json_path = json_file.split('.')[0] + '.json'
                json_path = os.path.join(vid_dir, json_path)
                if not os.path.exists(json_path):
                    print(f"{json_path} not exists, skipping")
                    continue

                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                rf_path = json_data.get(RAW_FRAME_TYPE, "")
                lf_path = json_data.get(LABEL_FRAME_TYPE, "")
                rf_path = os.path.join(self.base_image_dir, rf_path)
                lf_path = os.path.join(self.base_image_dir, lf_path)

                question = json_data.get(QUESTION_DIR, "")
                answer = json_data.get(ANSWER_DIR, "")

                sep = answer.find('Reason')
                if sep == -1:
                    continue

                if all([os.path.isfile(rf_path), os.path.isfile(lf_path)]) and len(question) > 0 and len(answer) > 0:
                    rf_path_list.append(rf_path)
                    lf_path_list.append(lf_path)
                    question_list.append(question)
                    answer_list.append(answer)
                else:
                    print(f'Warning: {rf_path} or {lf_path} is not found, or {question} or {answer} is empty.')

        assert len(rf_path_list) == len(lf_path_list) == len(question_list) == len(answer_list)
        print(f'Total of samples is {len(rf_path_list)}')
        return rf_path_list, lf_path_list, question_list, answer_list

    def __getitem__(self, idx):
        images, labels, questions, answers = self.images, self.labels, self.questions, self.answers
        if not self.eval_only:  # When evaluating, random sampling is not used
            idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]

        image = cv2.imread(image_path)
        if self.dataset != 'LBW':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(  # (2,3,224,224)
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = torch.from_numpy(image).permute(2,0,1)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # label = np.expand_dims(label, axis=0)
        label = self.map_transform(label)

        question = questions[idx]
        answer = answers[idx]
        answer = random.choice(self.answer_list) + " {}".format(answer)

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n ' + question)
        conv.append_message(conv.roles[1], answer)

        conversation = [conv.get_prompt()]

        return (
            image_path,
            label_path,
            image,
            image_clip,
            label,
            question,
            answer,
            conversation,
        )
