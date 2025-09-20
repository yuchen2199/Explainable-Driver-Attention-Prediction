import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
from audtorch.metrics.functional import pearsonr
from collections import deque
from collections import OrderedDict
from transformers import BitsAndBytesConfig, CLIPVisionModel
from .llava.mm_utils import tokenizer_image_token
from .llava import conversation as conversation_lib

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, BLANK_QUESTION)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
import random


def CC(pred: torch.Tensor, gt: torch.Tensor, eps=1e-7):
    a = pearsonr(pred.flatten(), gt.flatten())
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    a = a.mean()
    pearson = float(a)
    return pearson


def KLDivergence(pred: torch.Tensor, gt: torch.Tensor, eps=1e-7):
    P = pred
    P = P / (eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
    Q = gt
    Q = Q / (eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))

    R = Q * torch.log(eps + Q / (eps + P))
    R = R.sum()
    kld = float(R)
    return kld

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map

def SIM(s_map, gt):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    s_map = s_map.squeeze(1)
    gt = gt.squeeze(1)
    batch_size = s_map.size(0)
    h = s_map.size(1)
    w = s_map.size(2)

    s_map_norm = normalize_map(s_map)
    gt_norm = normalize_map(gt)

    sum_s_map = torch.sum(s_map_norm.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

    assert expand_s_map.size() == s_map_norm.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

    s_map_norm = s_map_norm / (expand_s_map * 1.0)
    gt_norm = gt / (expand_gt * 1.0)

    s_map_norm = s_map_norm.view(batch_size, -1)
    gt_norm = gt_norm.view(batch_size, -1)
    # return torch.mean(torch.sum(torch.min(s_map, gt), 1))
    return torch.sum(torch.min(s_map_norm, gt_norm), 1)


def discretize_gt(gt, threshold=0.7):
    gt = gt.astype(np.float32)
    epsilon = 1e-6
    binary_gt = np.where(gt >= threshold - epsilon, 1.0, 0.0)
    assert np.isin(binary_gt, [0, 1]).all(), "discretize error"
    return binary_gt


def AUC_J(s_map, gt):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    # gt = gt[0].squeeze(0).cpu().detach().numpy()
    # ground truth is discrete, s_map is continous and normalized
    gt = discretize_gt(gt)
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        assert np.max(gt) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(s_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
    # tp_list.append(tp)
    # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def AUC_B(s_map, gt, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map.squeeze(1))
    s_map = s_map[0].cpu().detach().numpy()
    gt = normalize_map(gt.squeeze(1))
    gt = gt[0].cpu().detach().numpy()
    # gt = gt[0].squeeze(0).cpu().detach().numpy()
    gt = discretize_gt(gt)
    num_fixations = np.sum(gt)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        for k in range(0, int(num_fixations)):
            temp_list.append(np.random.randint(num_pixels))
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def AUC_S(s_map, gt, other_map, splits=100, stepsize=0.1):
    # gt = discretize_gt(gt)
    # other_map = discretize_gt(other_map)

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def NSS(s_map, gt):
    s_map = s_map[0].squeeze(0).cpu().detach().numpy()
    gt = gt[0].squeeze(0).cpu().detach().numpy()
    gt = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)

    x, y = np.where(gt == 1)
    temp = []
    for i in zip(x, y):
        temp.append(s_map_norm[i[0], i[1]])
    return np.mean(temp)
    # MAP = (s_map - s_map.mean()) / (s_map.std())
    # mask = gt.astype(np.bool_)
    #
    # score =  MAP[mask].mean()
    # return score

def sal_bce_loss(pred, gt):
    bce_loss = nn.BCELoss()
    # bce_loss = nn.BCEWithLogitsLoss()
    # pred = pred.sigmoid()
    # gt = gt.sigmoid()
    loss = bce_loss(pred, gt)
    return loss

def sal_mse_loss(pred, gt):
    bce_loss = nn.MSELoss()
    loss = bce_loss(pred, gt)
    return loss

def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    # c = s_map.size(1)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 1e-8
    result = gt * torch.log(eps + gt / (s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))
    # return result.reshape(batch_size, w, h)

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class AttnMetaModel:
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(AttnMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_attn_decoder"):
            self.config.train_attn_decoder = kwargs["train_attn_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_attn_modules(self.config)

    def initialize_attn_modules(self, config):
        # attention prediction decode
        self.attn_decoder = AttentionDecoder()
        if config.train_attn_decoder:
            self.attn_decoder.train()
            for param in self.attn_decoder.parameters():
                param.requires_grad = True
        # Projection layer
        in_dim = config.hidden_size     # 4096
        out_dim = 1024        # 256
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class AttnModel(AttnMetaModel, LlavaLlamaModel):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(AttnModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower  # "openai/clip-vit-large-patch14"
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class AttnForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        if not hasattr(config, "train_attn_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.attn_loss_weight = kwargs.pop("attn_loss_weight", None)
            self.ce_what_loss_weight = kwargs.pop("ce_what_loss_weight", None)
            self.ce_why_loss_weight = kwargs.pop("ce_why_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.attn_token_idx = kwargs.pop("attn_token_idx")

        super().__init__(config)

        self.model = AttnModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):  # (1,3,1024,1024)
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(  # (1,256, 64, 64)
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        elif "pcsd_eval" in kwargs:
            return self.pcsd_evaluate(**kwargs)
        return self.model_forward(**kwargs)


    def model_forward(
            self,
            # images: torch.FloatTensor,        # (2,3,720,1280)
            images_clip: torch.FloatTensor,     # (2,3,224,224)
            input_ids: torch.LongTensor,        # (2,447)
            labels: torch.LongTensor,           # (2,447)
            labels_what: torch.LongTensor,      # (2,447)
            labels_why: torch.LongTensor,       # (2,447)
            gt_salmap: torch.FloatTensor,       # (2,1,256,256)
            attention_masks: torch.LongTensor,
            eval_text: bool = False,
            eval_only: bool = False,
            tokenizer=None,
            inference: bool = False,
            **kwargs,
    ):

        batch_size = images_clip.shape[0]

        if not eval_text:
            attn_token_mask = input_ids[:, 1:] == self.attn_token_idx  # (2, 446)
            attn_token_mask = torch.cat(  # (2, 447)
                [
                    attn_token_mask,
                    torch.zeros((attn_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            attn_token_mask = torch.cat(  # （2，702） 702=255+447
                [torch.zeros((attn_token_mask.shape[0], 255)).bool().cuda(), attn_token_mask],
                dim=1,
            )

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                labels_what=labels_what,
                labels_why=labels_why,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
            output_image_features = output.image_features
            output_ids = output.output_ids

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](output_hidden_states[-1]))  # 将最后一层隐藏状态进行mlp投影到attn decoder? (2,702,1024)
            if inference:
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1).unsqueeze(0)  # (2,702,256)
            else:
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # random_index = torch.randint(0, last_hidden_state.shape[1], (1,)).item()
            # pred_embeddings = last_hidden_state[:, random_index, :]
            pred_embeddings = last_hidden_state[attn_token_mask]  # 从最后一个隐藏层特征中提取出[ATTN]对应特征 (2,256)

            gt_sal = gt_salmap.to(self.device)
            pred_sal = self.model.attn_decoder(output_image_features, pred_embeddings)      # (2,256,4096) (2,256)
            assert pred_sal.shape == gt_sal.shape

            blur_func = transforms.GaussianBlur(11, 2)
            pred_sal = blur_func(pred_sal)
            pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)

        if eval_only:
            attn_loss, ce_loss, ce_what_loss, ce_why_loss, loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        else:
            kld_loss = kldiv(pred_sal.squeeze(1).to(gt_sal.dtype), gt_sal.squeeze(1)) * 0.1
            bce_loss = sal_bce_loss(pred_sal.to(gt_sal.dtype), gt_sal) * 1.0
            attn_loss = kld_loss * self.attn_loss_weight + bce_loss * self.attn_loss_weight

            model_output = output

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight

            ce_what_loss = model_output.loss_what * self.ce_what_loss_weight
            ce_why_loss = model_output.loss_why * self.ce_why_loss_weight

            if torch.isnan(ce_why_loss).any():
                print('loss_why is nan, so output the conv for checking...')
                conversation_list = kwargs['conversation_list']
                for i, conv in enumerate(conversation_list):
                    print(f'{i + 1}. {conv}')

            # loss = ce_loss + attn_loss
            loss = attn_loss + ce_what_loss + ce_why_loss

        if eval_only and eval_text:
            conv = conversation_lib.conv_templates['llava_v1'].copy()
            conv.messages = []
            question = kwargs['questions_list'][0]
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
            if True:
                replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                question = question.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], "")
            question = conv.get_prompt()
            input_ids = tokenizer_image_token(question, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            with torch.no_grad():
                text_output = self.generate(
                    images=images_clip,
                    input_ids=input_ids,
                    max_new_tokens=256,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                output_ids = text_output.sequences  # (1,54)
                output_hidden_states = text_output.hidden_states[-1]
                attn_token_mask = output_ids[:, 1:] == self.attn_token_idx
                # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
                attn_token_mask = torch.cat(
                    [
                        torch.zeros((attn_token_mask.shape[0], 255)).bool().cuda(),
                        attn_token_mask,
                    ],
                    dim=1,
                )

                hidden_states = []

                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(
                    self.model.text_hidden_fcs[0](output_hidden_states))

                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = last_hidden_state[attn_token_mask]  # (1, 256)
                image_features = self.encode_images(images_clip)
                pred_sal = self.model.attn_decoder(image_features, pred_embeddings)
                gt_sal = gt_salmap.to(self.device)
                assert pred_sal.shape == gt_sal.shape

                blur_func = transforms.GaussianBlur(11, 2)
                pred_sal = blur_func(pred_sal)
                pred_sal = transforms.Resize(gt_sal.shape[-2:])(pred_sal)
        else:
            output_ids = output_ids

        if inference:
            return {
                "pred_sal": pred_sal,
                "gt_sal": gt_sal,
                "output_ids": output_ids,
                "loss": loss,
                "ce_loss": ce_loss,
                "ce_what_loss": ce_what_loss,
                "ce_why_loss": ce_why_loss,
                "attn_loss": attn_loss,
            }

        return {
            "pred_sal": pred_sal,
            "gt_sal": gt_sal,
            "loss": loss,
            "ce_loss": ce_loss,
            "ce_what_loss": ce_what_loss,
            "ce_why_loss": ce_why_loss,
            "attn_loss": attn_loss,
        }


    def evaluate(
            self,
            images_clip,
            input_ids,
            original_size_list,
            max_new_tokens=32,
            tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]  # (1,308,5120)
            output_ids = outputs.sequences  # (1,54)

            attn_token_mask = output_ids[:, 1:] == self.attn_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            attn_token_mask = torch.cat(
                [
                    torch.zeros((attn_token_mask.shape[0], 255)).bool().cuda(),
                    attn_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[attn_token_mask]
            image_features = self.encode_images(images_clip)
            pred_sal = self.model.attn_decoder(image_features, pred_embeddings)

        return output_ids, pred_sal



class Decoder_ConvBlock(nn.Module):
    """ special case for Convblock with up sampling zhouyc
    """
    def __init__(self, inplanes, med_planes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Decoder_ConvBlock, self).__init__()


        # self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=2)

        self.drop_block = drop_block
        self.drop_path = drop_path


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        # if self.drop_block is not None:
        #     x = self.drop_block(x)
        x = self.act1(x)
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = self.upsample_layer(x)
        if dtype == torch.bfloat16:
            x = x.to(torch.bfloat16)


        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V):
        # Q: (bs, 1, embed_dim), K: (bs, H*W, embed_dim), V: (bs, H*W, embed_dim)
        attn_output, _ = self.attention(Q, K, V)  # (bs, 1, embed_dim)
        Q = Q + attn_output  # Residual connection
        Q = self.norm(Q)
        mlp_output = self.mlp(Q)  # (bs, 1, embed_dim)
        Q = Q + mlp_output  # Residual connection
        Q = self.norm2(Q)
        return Q


class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=3, output_layer=6):
        super(AttentionDecoder, self).__init__()
        self.output_layer = output_layer
        self.embed_dim = embed_dim
        self.visual_proj = nn.Linear(4096, 1024)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # decoder
        self.output_layer = output_layer
        self.decoder1 = Decoder_ConvBlock(1024, 512)
        self.decoder2 = Decoder_ConvBlock(512, 256)
        self.decoder3 = Decoder_ConvBlock(256, 64)
        self.decoder4 = Decoder_ConvBlock(64, 32)
        self.readout = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, visual_features, llm_hidden_state):
        bs, num_patches, _ = visual_features.size()
        visual_features = self.visual_proj(visual_features)  # (bs, 256, 1024)

        # Prepare visual features as key (K) and value (V)
        key = visual_features  # (bs, 256, 1024)
        value = visual_features  # (bs, 256, 1024)


        # Prepare LLM hidden state as query (Q)
        query = llm_hidden_state.unsqueeze(1)  # (bs, 1, 1024)

        # Pass through cross-attention layers
        for layer in self.cross_attention_layers:
            query = layer(query, key, value)  # (bs, 1, 1024)

        query = query.repeat(1, num_patches, 1)  # (bs, 256, 1024)
        fused_features = query + visual_features  # (bs, 256, 1024)

        H = W = int(num_patches ** 0.5)
        x = fused_features.permute(0, 2, 1).view(bs, self.embed_dim, H, W)  # (bs, 1024, H, W) H=W=16
        # Pass through convolutional decoder
        x = self.decoder1(x)  # (bs, 128, H*2, W*2)
        x = self.decoder2(x)  # (bs, 64, H*4, W*4)
        x = self.decoder3(x)  # (bs, 32, H*8, W*8)
        y = self.decoder4(x)  # (bs, 1, H*16, W*16)
        y = self.readout(y)
        return y

