import argparse
import os
import re
import sys

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.Attn_model import AttnForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LLaDA Chat")
    parser.add_argument("--version", default="./ATTN-03012102-new-epoch4")
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
    parser.add_argument(
        "--vision-tower", default=r"./weights/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
tokenizer = AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
args.attn_token_idx = tokenizer("[ATTN]", add_special_tokens=False).input_ids[0]

torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half

kwargs = {"torch_dtype": torch_dtype}
if args.load_in_4bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    )
elif args.load_in_8bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        }
    )

model = AttnForCausalLM.from_pretrained(
    args.version,
    low_cpu_mem_usage=True,
    vision_tower=args.vision_tower,
    attn_token_idx=args.attn_token_idx,
    **kwargs
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif (
    args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
):
    vision_tower = model.get_model().get_vision_tower()
    model.model.vision_tower = None
    import deepspeed

    model_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.half,
        replace_with_kernel_inject=True,
        replace_method="auto",
    )
    model = model_engine.module
    model.model.vision_tower = vision_tower.half().cuda()
elif args.precision == "fp32":
    model = model.float().cuda()

vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank)

clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

model.eval()


# Gradio
output_labels = ["Attention Prediction Output"]

title = "Where, What, Why: Towards Explainable Driver Attention Prediction"

description = """
<font size=4>
This is the online demo of LLaDA. \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Usage**: <br> Input Prompt Like: 
Imagine you are driving in the provided image. The driving scenario is as follows: <OPTIONAL CONTEXT>. The image shows your visual attention distribution. 
Please answer the following questions briefly: 
1. How many regions of attention are present in the image? 
- Format your response as: 'Number of regions: [number]' 
2. What are the specific regions where the driver’s attention is focused? 
- For each region, format your response as follows: 
    - 'Region 1: [Name of the first region]' 
    - (If applicable) 'Region 2: [Name of the second region]' 
    - (If applicable) 'Region 3: [Name of the third region]' 
3. Why is the driver focusing on these regions? 
- For each reason, consider the impact on upcoming driving decisions, start with "To ...", and format your response as follows: 
    - 'Reason 1: [Explanation for the first region]' 
    - (If applicable) 'Reason 2: [Explanation for the second region]' 
    - (If applicable) 'Reason 3: [Explanation for the third region]' 
Important Notes for Your Response: - Avoid assumptions about elements not visible in the image (e.g., side mirrors or rearview mirrors or dashboard or control panel). 
 & <br>
Hope you can enjoy our work!
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/xxxx.xxxxx' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/xxxxxx' target='_blank'>   Github Repo </a></p>
"""

def visualize_heatmap(image, gazemap):
    gazemap = cv2.normalize(gazemap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = image.shape[:2]
    resized_pred = cv2.resize(gazemap, (w, h))

    resized_pred = cv2.applyColorMap(resized_pred, cv2.COLORMAP_JET)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blended_image = cv2.addWeighted(image, 0.5, resized_pred, 0.5, 0)
    return blended_image


## to be implemented
def inference(input_str, input_image):
    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    ## input valid check
    if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
        output_str = "[Error] Invalid input: ", input_str
        # output_image = np.zeros((128, 128, 3))
        ## error happened
        output_image = cv2.imread("./resources/error_happened.png")[:, :, ::-1]
        return output_image, output_str

    # Model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = input_str
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_sal = model.evaluate(
        image_clip,
        input_ids,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    # text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split('ASSISTANT: ')[-1][:-4]

    print("text_output: ", text_output)

    output_str = "ASSITANT: " + text_output  # input_str

    pred_np = pred_sal[0].cpu().detach().to(torch.half).numpy()
    pred_np = (pred_np * 255).astype(np.uint8)
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
    pred_np = np.squeeze(pred_np)

    pred_heatmap = visualize_heatmap(image_np, pred_np)
    output_image = pred_heatmap  # input_image
    return output_image, output_str


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Image(type="pil", label="Saliency Map Output"),
        gr.Textbox(lines=1, placeholder=None, label="Text Output"),
    ],
    title=title,
    description=description,
    article=article,
    # examples=examples,
    allow_flagging="auto",
)

demo.queue()
demo.launch(share=True)
