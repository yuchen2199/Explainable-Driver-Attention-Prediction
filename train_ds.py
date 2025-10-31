import argparse
import os

# os.environ['TORCH_USE_CUDA_DSA'] = '1'
import shutil
import sys
import time
import cv2
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
from datetime import datetime
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from model.Attn_model import AttnForCausalLM, CC, KLDivergence, SIM, NSS, AUC_J, AUC_B
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         IMAGE_TOKEN_INDEX)

from utils.eval_utils.bleu import Bleu
from utils.eval_utils.cider import Cider
from utils.eval_utils.ciderR import CiderR
from utils.eval_utils.meteor.meteor import Meteor
from utils.eval_utils.rouge import Rouge

def parse_args(args):
    parser = argparse.ArgumentParser(description="LLaDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="./weights/LLaVA-7B-Lightening-v1-1"   # Path to pretrained LLaVA model, or already trained LLada model
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--map_size", default=256, type=int)
    parser.add_argument(
        "--vision-tower", default=r"./weights/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="BDDA||DReyeVE||LBW||DADA", type=str
    )
    parser.add_argument("--val_dataset", default="BDDA||DReyeVE||LBW||DADA", type=str)
    parser.add_argument("--train_sample_rates", default="8,5,2,7", type=str)
    parser.add_argument("--val_sample_rates", default="8,5,2,7", type=str)
    parser.add_argument("--dataset_dir", default="./dataset_keyframes", type=str)
    parser.add_argument("--log_base_dir", default="./ATTN/runs", type=str)
    parser.add_argument("--exp_name", default="Attn", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--val_samples_num", default=5000, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--attn_loss_weight", default=2.0, type=float)
    parser.add_argument("--ce_what_loss_weight", default=1.0, type=float)
    parser.add_argument("--ce_why_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--eval_text", action="store_true", default=False)
    parser.add_argument("--eval_text_resume", default=None)
    parser.add_argument("--eval_text_save", default=False)
    parser.add_argument("--eval_colormap_save", default=False)
    parser.add_argument("--out_dim", default=1024, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_attn_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[ATTN]")    # added_tokens_decoder/encoder
    args.attn_token_idx = tokenizer("[ATTN]", add_special_tokens=False).input_ids[0]    # 32003

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_attn_decoder": args.train_attn_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "attn_loss_weight": args.attn_loss_weight,
        "ce_what_loss_weight": args.ce_what_loss_weight,
        "ce_why_loss_weight": args.ce_why_loss_weight,
        "attn_token_idx": args.attn_token_idx,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = AttnForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_attn_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "attn_decoder",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha            # 16
        lora_dropout = args.lora_dropout        # 0.05
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")  # q_proj， v_proj
        )
        lora_config = LoraConfig(
            r=lora_r,                   # 8
            lora_alpha=lora_alpha,      # 16
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,  # 0.05
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, attn_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "attn_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.world_size = world_size
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        map_size=args.map_size,
        precision=args.precision,
        image_size=args.image_size,
        split='training',
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.train_sample_rates.split(",")],
        eval_only=False,
    )

    if args.no_eval == False:
        val_dataset = HybridDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=args.val_samples_num,
            precision=args.precision,
            image_size=args.image_size,
            split='test',
            dataset=args.val_dataset,
            sample_rate=[float(x) for x in args.val_sample_rates.split(",")],
            eval_only=args.eval_only,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        # assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    # best_score, cur_what_l, cur_why_l = np.inf, 0.0, 0.0
    best_kld, best_cc, best_sim, best_nss, best_aucb, best_aucj, cur_what_l, cur_why_l, cur_attn_l, cur_l, cur_kld, cur_cc, cur_sim, cur_nss, cur_aucb, cur_aucj = np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if args.eval_only and args.eval_text:
        args.log_dir = os.path.join(args.log_dir, 'text_eval')
        args.log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    elif args.eval_only:
        args.log_dir = os.path.join(args.log_dir, 'attn_eval')
        args.log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    elif args.eval_text_resume is None:
        args.log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    else:
        args.log_dir = args.eval_text_resume
    os.makedirs(args.log_dir, exist_ok=True)

    config_text = f"model_version: {args.version}, eval_dataset: {args.val_dataset}, dataset_sample_rate: {args.val_sample_rates}, sample_total: {len(val_dataset)}, gpu_num: {world_size}\n"
    if args.eval_only:
        loss, attn_loss, text_loss, what_loss, why_loss, cc, kld, sim, nss, auc_b, auc_j, comp_tm, what_tm, why_tm = validate(val_loader, model_engine, 0, writer, args, tokenizer, eval_text=args.eval_text, eval_only=True, save_vis=True)
        bleu, meteor, rouge, cider, ciderR = comp_tm
        bleu_wt, meteor_wt, rouge_wt, cider_wt, ciderR_wt = what_tm
        bleu_wy, meteor_wy, rouge_wy, cider_wy, ciderR_wy = why_tm

        log_text = f"{config_text}The evaluation result: \n"
        log_text += f"Attn metrics:\nCC: {cc:.6f}, KLD: {kld:.6f}, SIM: {sim:.6f}, NSS: {nss:.6f}, AUC_B: {auc_b:.6f}, AUC_J: {auc_j:.6f}\n"
        log_text += f"Text metrics (Complete):\nBleu_4: {bleu[3]:.6f}, Bleu_3: {bleu[2]:.6f}, Bleu_2: {bleu[1]:.6f}, Bleu_1: {bleu[0]:.6f}, Meteor: {meteor:.6f}, Rouge: {rouge:.6f}, Cider: {ciderR:.6f}, CiderR: {ciderR:.6f}\n"
        log_text += f"Text metrics (What part):\nBleu_4: {bleu_wt[3]:.6f}, Bleu_3: {bleu_wt[2]:.6f}, Bleu_2: {bleu_wt[1]:.6f}, Bleu_1: {bleu_wt[0]:.6f}, Meteor: {meteor_wt:.6f}, Rouge: {rouge_wt:.6f}, Cider: {ciderR_wt:.6f}, CiderR: {ciderR_wt:.6f}\n"
        log_text += f"Text metrics (Why part):\nBleu_4: {bleu_wy[3]:.6f}, Bleu_3: {bleu_wy[2]:.6f}, Bleu_2: {bleu_wy[1]:.6f}, Bleu_1: {bleu_wy[0]:.6f}, Meteor: {meteor_wy:.6f}, Rouge: {rouge_wy:.6f}, Cider: {ciderR_wy:.6f}, CiderR: {ciderR_wy:.6f}"
        save_path = os.path.join(args.log_dir, "log_test.txt")
        with open(save_path, "w") as f:
            f.write(log_text)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            loss, attn_loss, text_loss, what_loss, why_loss, cc, kld, sim, nss, auc_b, auc_j, comp_tm, what_tm, why_tm = validate(val_loader, model_engine, epoch, writer, args, tokenizer, eval_text=False, eval_only=False, save_vis=True)
            # is_best = attn_loss + text_loss < best_score
            # best_score = min(attn_loss + text_loss, best_score)
            is_best = kld < best_kld or cc > best_cc or sim > best_sim or nss > best_nss or auc_b > best_aucb or auc_j > best_aucj
            best_kld = min(kld, best_kld)
            best_cc = max(cc, best_cc)
            best_sim = max(sim, best_sim)
            best_nss = max(nss, best_nss)
            best_aucb = max(auc_b, best_aucb)
            best_aucj = max(auc_j, best_aucj)
            cur_kld = kld
            cur_cc = cc
            cur_sim = sim
            cur_nss = nss
            cur_aucb = auc_b
            cur_aucj = auc_j
            cur_what_l = what_loss
            cur_why_l = why_loss
            cur_attn_l = attn_loss
            cur_l = loss

            log_text = f"{config_text}\nEpoch{epoch} validation metrics: \n"
            log_text += f"KLD:{kld:.6f}, CC:{cc:.6f}, SIM:{sim:.6f}, NSS:{nss:.6f}, AUC_B:{auc_b:.6f}, AUC_J:{auc_j:.6f}\n"
            log_text += f"Attn_Loss: {attn_loss:.6f}, Total_Loss: {loss:.6f}, What_Loss: {what_loss:.6f}, Why_Loss: {why_loss:.6f}"

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, f"best_ckpt_model_epoch{epoch}")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_tl{:.3f}_wtl{:.3f}_wyl{:.3f}_al{:.3f}_sim{:.3f}_cc{:.3f}_kld{:.3f}_nss{:.3f}_aucb{:.3f}_aucj{:.3f}_best.pth".format(
                            cur_l, cur_what_l, cur_why_l, cur_attn_l, cur_sim, cur_cc, cur_kld, cur_nss, cur_aucb, cur_aucj
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            log_save = os.path.join(save_dir, 'log.txt')
            with open(log_save, 'w') as f:
                f.write(log_text)
            print('Saving best checkpoint to', save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    ce_what_losses = AverageMeter("CeWhatLoss", ":.4f")
    ce_why_losses = AverageMeter("CeWhyLoss", ":.4f")
    attn_losses = AverageMeter("AttnLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            ce_what_losses,
            ce_why_losses,
            attn_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    j = 0
    for global_step in range(args.steps_per_epoch):     # 500
        for i in range(args.grad_accumulation_steps):   # 10
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            pred_sal = output_dict["pred_sal"]
            gt_sal = output_dict["gt_sal"]
            if j % 100 == 0:
                vis_save = os.path.join(args.log_dir, 'train_vis')
                os.makedirs(vis_save, exist_ok=True)
                epoch_save = os.path.join(vis_save, f'epoch_{epoch}')
                os.makedirs(epoch_save, exist_ok=True)
                sal_save = os.path.join(epoch_save, f'pred_{j}.jpg')
                gt_save = os.path.join(epoch_save, f'gt_{j}.jpg')
                save_salmap(pred_sal, sal_save)
                save_salmap(gt_sal, gt_save)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            ce_what_loss = output_dict["ce_what_loss"]
            ce_why_loss = output_dict["ce_why_loss"]
            attn_loss = output_dict["attn_loss"]

            losses.update(loss.item(), input_dict["images_clip"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images_clip"].size(0))
            ce_what_losses.update(ce_what_loss.item(), input_dict["images_clip"].size(0))
            ce_why_losses.update(ce_why_loss.item(), input_dict["images_clip"].size(0))
            attn_losses.update(attn_loss.item(), input_dict["images_clip"].size(0))
            model.backward(loss)
            model.step()

            j += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                ce_what_losses.all_reduce()
                ce_why_losses.all_reduce()
                attn_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar("train/ce_what_losses", ce_what_losses.avg, global_step)
                writer.add_scalar("train/ce_why_losses", ce_why_losses.avg, global_step)
                writer.add_scalar("train/attn_loss", attn_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            ce_what_losses.reset()
            ce_why_losses.reset()
            attn_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def save_salmap(sal, save_path):
    sal = sal.to(torch.half)
    sal = sal[0].squeeze(0).cpu().detach().numpy()
    sal = (sal * 255).astype(np.uint8)
    sal = np.clip(sal, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, sal)

def save_txt(txt, save_path):
    with open(save_path, 'w') as f:
        f.write(txt)

def save_txt2dataset(txt, image_path, log_dir):
    log_name = log_dir.split('/')[-1]
    image_name = image_path.split('/')[-1].split('.')[0]
    vid_dir = image_path.split('raw_frames')[0]
    eval_dir = os.path.join(vid_dir, 'eval_text')
    os.makedirs(eval_dir, exist_ok=True)
    log_path = os.path.join(eval_dir, log_name)
    os.makedirs(log_path, exist_ok=True)
    save_path = os.path.join(log_path, image_name + '.txt')
    with open(save_path, 'w') as f:
        f.write(txt)

def sep_what_and_why(txt):
    what_end = txt.find("3. Reason")
    if what_end == -1:
        what_end = txt.find("- Reason")
    if what_end == -1:
        what_end = txt.find("Reason")
    if what_end == -1:
        what_end = txt.find("3.")
    what_part = txt[:what_end] if what_end != -1 else txt
    why_part = txt[what_end:] if what_end != -1 else ""
    return what_part, why_part


def save_colormap(image_path, sal, log_dir):
    sal = sal.to(torch.half)
    sal = sal[0].squeeze(0).cpu().detach().numpy()
    sal = (sal * 255).astype(np.uint8)
    cur_image = cv2.imread(image_path)
    gazemap = cv2.resize(sal, (cur_image.shape[1], cur_image.shape[0]))
    gazemap_norm = cv2.normalize(gazemap, None, 0, 255, cv2.NORM_MINMAX)
    colormap_gazemap = cv2.applyColorMap(gazemap_norm, cv2.COLORMAP_JET)
    colormap = cv2.addWeighted(cur_image, 0.8, colormap_gazemap, 0.2, 0)

    log_name = log_dir.split('/')[-1]
    image_name = image_path.split('/')[-1].split('.')[0]
    vid_dir = image_path.split('raw_frames')[0]
    eval_dir = os.path.join(vid_dir, 'eval_saving')
    os.makedirs(eval_dir, exist_ok=True)
    log_path = os.path.join(eval_dir, log_name)
    os.makedirs(log_path, exist_ok=True)
    save_path = os.path.join(log_path, image_name + '.jpg')
    # cv2.imshow('colormap', colormap)
    # cv2.waitKey(0)
    cv2.imwrite(save_path, colormap)


def validate(val_loader, model_engine, epoch, writer, args, tokenizer, eval_text=False, eval_only=True, save_vis=False):
    cur_local_rank = dist.get_rank() % torch.cuda.device_count()

    loss_meter = AverageMeter("Loss", ":6.3f", Summary.SUM)
    attn_loss_meter = AverageMeter("AttnLoss", ":6.3f", Summary.SUM)
    text_loss_meter = AverageMeter("TextLoss", ":6.3f", Summary.SUM)
    what_loss_meter = AverageMeter("WhatLoss", ":6.3f", Summary.SUM)
    why_loss_meter = AverageMeter("WhyLoss", ":6.3f", Summary.SUM)
    cc_meter = AverageMeter("CC", ":6.3f", Summary.SUM)
    kld_meter = AverageMeter("KLDivergence", ":6.3f", Summary.SUM)
    sim_meter = AverageMeter("SIM", ":6.3f", Summary.SUM)
    nss_meter = AverageMeter("NSS", ":6.3f", Summary.SUM)
    aucb_meter = AverageMeter("AUC_B", ":6.3f", Summary.SUM)
    aucj_meter = AverageMeter("AUC_J", ":6.3f", Summary.SUM)

    bleu_metric = Bleu(4)
    meteor_metric = Meteor()
    rouge_metric = Rouge()
    cider_metric = Cider()
    ciderR_metric = CiderR()

    bleu_metric_wt = Bleu(4)
    meteor_metric_wt = Meteor()
    rouge_metric_wt = Rouge()
    cider_metric_wt = Cider()
    ciderR_metric_wt = CiderR()

    bleu_metric_wy = Bleu(4)
    meteor_metric_wy = Meteor()
    rouge_metric_wy = Rouge()
    cider_metric_wy = Cider()
    ciderR_metric_wy = CiderR()

    # record the attn metric of each sample
    attn_sample_metrics = []

    model_engine.eval()
    i = 0

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images_clip"] = input_dict["images_clip"].float()

        input_dict["eval_text"] = eval_text
        input_dict["eval_only"] = eval_only
        input_dict["tokenizer"] = tokenizer
        image_path = input_dict['image_paths'][0]

        # checking whether to resume the text eval
        if eval_text and args.eval_text_resume is not None:
            log_name = args.eval_text_resume
            image_name = image_path.split('/')[-1].split('.')[0]
            vid_dir = image_path.split('raw_frames')[0]
            eval_dir = os.path.join(vid_dir, 'eval_text')
            log_path = os.path.join(eval_dir, log_name)
            text_path = os.path.join(log_path, image_name + '.txt')
            if os.path.exists(text_path):
                gt_text = input_dict["answers_list"][0]
                with open(text_path, 'r') as f:
                    pred_text = f.read()
                gt_what, gt_why = sep_what_and_why(gt_text)
                pred_what, pred_why = sep_what_and_why(pred_text)

                bleu_metric.append([gt_text], pred_text)
                meteor_metric.append([gt_text], pred_text)
                rouge_metric.append([gt_text], [pred_text])
                cider_metric.append([gt_text], pred_text)
                ciderR_metric.append([gt_text], pred_text)

                bleu_metric_wt.append([gt_what], pred_what)
                meteor_metric_wt.append([gt_what], pred_what)
                rouge_metric_wt.append([gt_what], [pred_what])
                cider_metric_wt.append([gt_what], pred_what)
                ciderR_metric_wt.append([gt_what], pred_what)

                bleu_metric_wy.append([gt_why], pred_why)
                meteor_metric_wy.append([gt_why], pred_why)
                rouge_metric_wy.append([gt_why], [pred_why])
                cider_metric_wy.append([gt_why], pred_why)
                ciderR_metric_wy.append([gt_why], pred_why)

                print(f'{text_path} already exists, without generation again')
                continue

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_sal = output_dict["pred_sal"]
        gt_sal = output_dict["gt_sal"]
        # calculate the attn metrics
        if pred_sal is None or gt_sal is None:
            cc, kld, sim, nss, auc_b, auc_j = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            cc = CC(pred_sal.to(gt_sal.dtype), gt_sal)
            kld = KLDivergence(pred_sal.to(gt_sal.dtype), gt_sal)
            sim = SIM(pred_sal.to(gt_sal.dtype), gt_sal)
            nss = NSS(pred_sal.to(gt_sal.dtype), gt_sal)
            auc_j = AUC_J(pred_sal.to(gt_sal.dtype), gt_sal)
            auc_b = AUC_B(pred_sal.to(gt_sal.dtype), gt_sal)

            # prepare the text metrics
        if eval_text:
            output_ids = output_dict['output_ids']

            pts = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            pred_text = tokenizer.decode(pts[:-1], skip_special_tokens=False)
            pred_text = pred_text.split('ASSISTANT: ')[-1]
            pred_text = pred_text[:pred_text.find('</s>')]
            gt_text = input_dict["answers_list"][0]

            gt_what, gt_why = sep_what_and_why(gt_text)
            pred_what, pred_why = sep_what_and_why(pred_text)

            bleu_metric.append([gt_text], pred_text)
            meteor_metric.append([gt_text], pred_text)
            rouge_metric.append([gt_text], [pred_text])
            cider_metric.append([gt_text], pred_text)
            ciderR_metric.append([gt_text], pred_text)

            bleu_metric_wt.append([gt_what], pred_what)
            meteor_metric_wt.append([gt_what], pred_what)
            rouge_metric_wt.append([gt_what], [pred_what])
            cider_metric_wt.append([gt_what], pred_what)
            ciderR_metric_wt.append([gt_what], pred_what)

            bleu_metric_wy.append([gt_why], pred_why)
            meteor_metric_wy.append([gt_why], pred_why)
            rouge_metric_wy.append([gt_why], [pred_why])
            cider_metric_wy.append([gt_why], pred_why)
            ciderR_metric_wy.append([gt_why], pred_why)

        # saving the gt and pred saliency map
        if i % 300 == 0:
            if pred_sal is not None and gt_sal is not None:
                vis_save = os.path.join(args.log_dir, 'val_vis')
                os.makedirs(vis_save, exist_ok=True)
                epoch_save = os.path.join(vis_save, f'epoch_{epoch}')
                os.makedirs(epoch_save, exist_ok=True)
                sal_save = os.path.join(epoch_save, f'pred_{i}.jpg')
                gt_save = os.path.join(epoch_save, f'gt_{i}.jpg')
                save_salmap(pred_sal, sal_save)
                save_salmap(gt_sal, gt_save)

        # saving every heatmap
        if eval_only and i % 1 == 0 and args.eval_colormap_save:
            if pred_sal is not None and gt_sal is not None:
                save_colormap(image_path, pred_sal, args.log_dir)

        # saving every text output
        if eval_text and args.eval_text_save:
            save_txt2dataset(pred_text, image_path, args.log_dir)

        loss = output_dict["loss"]
        attn_loss = output_dict["attn_loss"]
        text_loss = output_dict["ce_loss"]
        what_loss = output_dict["ce_what_loss"]
        why_loss = output_dict["ce_why_loss"]

        loss_meter.update(loss.item())
        attn_loss_meter.update(attn_loss.item())
        text_loss_meter.update(text_loss.item())
        what_loss_meter.update(what_loss.item())
        why_loss_meter.update(why_loss.item())
        if not np.isnan(cc):
            cc_meter.update(cc)
        if not np.isnan(kld):
            kld_meter.update(kld)
        if not torch.isnan(sim):
            sim_meter.update(sim)
        if not np.isnan(nss):
            nss_meter.update(nss)
        if not np.isnan(auc_b):
            aucb_meter.update(auc_b)
        if not np.isnan(auc_j):
            aucj_meter.update(auc_j)

        global_idx = i * args.world_size + cur_local_rank
        attn_sample_metrics.append({
            'global_idx': global_idx,
            'local_idx': i,
            'rank': cur_local_rank,
            'image_path': image_path,
            'cc': float(cc),
            'kld': float(kld),
            'sim': float(sim),
            'nss': float(nss),
            'auc_b': float(auc_b),
            'auc_j': float(auc_j),
            'pred_text': pred_text if eval_text else None
        })

        i += 1

    loss_meter.all_reduce()
    attn_loss_meter.all_reduce()
    text_loss_meter.all_reduce()
    what_loss_meter.all_reduce()
    why_loss_meter.all_reduce()
    cc_meter.all_reduce()
    kld_meter.all_reduce()
    sim_meter.all_reduce()
    nss_meter.all_reduce()
    aucb_meter.all_reduce()
    aucj_meter.all_reduce()

    loss = loss_meter.avg
    attn_loss = attn_loss_meter.avg
    text_loss = text_loss_meter.avg
    what_loss = what_loss_meter.avg
    why_loss = why_loss_meter.avg
    # attn metrics
    cc = cc_meter.avg
    kld = kld_meter.avg
    sim = sim_meter.avg
    nss = nss_meter.avg
    auc_b = aucb_meter.avg
    auc_j = aucj_meter.avg
    # text metrics
    if eval_text:
        bleu, bleu_info = bleu_metric.compute_score()
        meteor, meteor_info = meteor_metric.compute_score()
        rouge, rouge_info = rouge_metric.compute_score()
        cider, cider_info = cider_metric.compute_score()
        ciderR, ciderR_info = ciderR_metric.compute_score()

        bleu_wt, _ = bleu_metric_wt.compute_score()
        meteor_wt, _ = meteor_metric_wt.compute_score()
        rouge_wt, _ = rouge_metric_wt.compute_score()
        cider_wt, _ = cider_metric_wt.compute_score()
        ciderR_wt, _ = ciderR_metric_wt.compute_score()

        bleu_wy, _ = bleu_metric_wy.compute_score()
        meteor_wy, _ = meteor_metric_wy.compute_score()
        rouge_wy, _ = rouge_metric_wy.compute_score()
        cider_wy, _ = cider_metric_wy.compute_score()
        ciderR_wy, _ = ciderR_metric_wy.compute_score()
    else:
        bleu, meteor, rouge, cider, ciderR = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0
        bleu_wt, meteor_wt, rouge_wt, cider_wt, ciderR_wt = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0
        bleu_wy, meteor_wy, rouge_wy, cider_wy, ciderR_wy = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0


    if args.local_rank == 0:
        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/attn_loss", attn_loss, epoch)
        writer.add_scalar("val/text_loss", text_loss, epoch)
        writer.add_scalar("val/what_loss", what_loss, epoch)
        writer.add_scalar("val/why_loss", why_loss, epoch)
        writer.add_scalar("val/cc", cc, epoch)
        writer.add_scalar("val/kld", kld, epoch)
        writer.add_scalar("val/sim", sim, epoch)
        writer.add_scalar("val/nss", nss, epoch)
        writer.add_scalar("val/auc_b", auc_b, epoch)
        writer.add_scalar("val/auc_j", auc_j, epoch)
        #
        writer.add_scalar("val/bleu_4", bleu[3], epoch)
        writer.add_scalar("val/bleu_3", bleu[2], epoch)
        writer.add_scalar("val/bleu_2", bleu[1], epoch)
        writer.add_scalar("val/bleu_1", bleu[0], epoch)
        writer.add_scalar("val/meteor", meteor, epoch)
        writer.add_scalar("val/rouge", rouge, epoch)
        writer.add_scalar("val/cider", cider, epoch)
        writer.add_scalar("val/ciderR", ciderR, epoch)

    # saving the attn metric of each sample
    metrics_path = os.path.join(args.log_dir, f'attn_metrics_{cur_local_rank}.csv')
    with open(metrics_path, 'w') as f:
        f.write("global_idx,local_idx,rank,image_id,cc,kld,sim,nss,auc_b,auc_j\n")
        for m in attn_sample_metrics:
            f.write(f"{m['global_idx']},{m['local_idx']},{m['rank']},"
                    f"{m['image_path']},{m['cc']:.6f},{m['kld']:.6f},{m['sim']:.6f},{m['nss']:.6f},{m['auc_b']:.6f},{m['auc_j']:.6f}\n")


    print("loss: {:.4f}, attn_loss: {:.4f}, text_loss: {:.4f}, what_loss: {:.4f}, why_loss: {:.4f}, cc: {:4f}, kld: {:4f}, sim: {:4f}, nss: {:4f}, auc_b: {:4f}, auc_j: {:4f}, "
              "Complete: bleu_4: {:4f}, bleu_3: {:4f}, bleu_2: {:4f}, bleu_1: {:4f}, meteor: {:4f}, rouge: {:4f}, cider: {:4f}, ciderR: {:4f}, "
              "What: bleu_4: {:4f}, bleu_6: {:4f}, bleu_7: {:4f}, bleu_1: {:4f}, meteor: {:4f}, rouge: {:4f}, cider: {:4f}, ciderR: {:4f}, "
              "Why: bleu_4: {:4f}, bleu_6: {:4f}, bleu_7: {:4f}, bleu_1: {:4f}, meteor: {:4f}, rouge: {:4f}, cider: {:4f}, ciderR: {:4f}"
              .format(loss, attn_loss, text_loss, what_loss, why_loss, cc, kld, sim, nss, auc_b, auc_j,
                      bleu[3], bleu[2], bleu[1], bleu[0], meteor, rouge, cider, ciderR,
                      bleu_wt[3], bleu_wt[2], bleu_wt[1], bleu_wt[0], meteor_wt, rouge_wt, cider_wt, ciderR_wt,
                      bleu_wy[3], bleu_wy[2], bleu_wy[1], bleu_wy[0], meteor_wy, rouge_wy, cider_wy, ciderR_wy))

    return (loss, attn_loss, text_loss, what_loss, why_loss, cc, kld, sim, nss, auc_b, auc_j,
            (bleu, meteor, rouge, cider, ciderR), (bleu_wt, meteor_wt, rouge_wt, cider_wt, ciderR_wt),
            (bleu_wy, meteor_wy, rouge_wy, cider_wy, ciderR_wy))


if __name__ == "__main__":
    main(sys.argv[1:])
