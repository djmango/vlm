from vit_pytorch.na_vit import NaViT
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
import time
import os

BS = 64
patch_size = 8
dim_head = 64
v = NaViT(
    image_size = 256,
    patch_size = 8,
    num_classes=1000,
    dim = 1024,
    depth = 12,
    heads = 2,
    mlp_dim = 2048
)

def train_time(toks, gpu_flops=4e12):
    # Calculate N (number of parameters)
    d_attn = v.dim // v.heads  # Attention head dimension
    dff = v.mlp_dim  # Feed-forward dimension
    N = 2 * v.dim * v.depth * (2 * d_attn + dff)
    # C = 6NB by rule of thumb, where N = number of parameters and B = batch size
    C = 6 * N * toks
    # Calculate training time in hours
    time_hr = (C / gpu_flops) / (60*60)
    return time_hr

def calc_inference_memory(v : NaViT, bs, seqlen, bytes=4):
    one_tok = v.heads * v.dim * dim_head * 3 # qkv
    total_attn = bs * seqlen * one_tok 
    return total_attn * bytes / (1024 ** 3)

def calc_model_memory():
    total_params = sum(p.numel() for p in v.parameters())
    memory_usage = total_params * 4 / (1024 ** 3)  # 4 bytes per parameter (fp32), convert to GB
    return memory_usage

# Print memory usage for different YouTube resolutions and calculate maximum batch size
print("Memory usage and maximum batch size for different resolutions:")

def calc_max_batch_size(memory_usage, available_memory=80):  # Assuming 24GB GPU
    return int(available_memory / memory_usage)

resolutions = [
    ("4K (3840x2160)", 3840, 2160),
    ("1440p (2560x1440)", 2560, 1440),
    ("1080p (1920x1080)", 1920, 1080),
    ("720p (1280x720)", 1280, 720),
    ("480p (854x480)", 854, 480),
    ("360p (640x360)", 640, 360),
    ("240p (426x240)", 426, 240)
]

dataset_rows = 1e6 
for name, width, height in resolutions:
    print(f"\n{name}:")
    memory_usage = calc_inference_memory(v, 1, )
    print(f"Memory usage: {memory_usage:.2f} GB")
    max_batch_size = calc_max_batch_size(memory_usage)
    print(f"Max batch size at 80GB memory: {max_batch_size}")
    num_total_batches = dataset_rows / max_batch_size
    toks_per_batch = (width // patch_size) * (height // patch_size) * max_batch_size
    total_toks = toks_per_batch * num_total_batches
    print(f"H100 hours to train batch: {train_time(toks_per_batch)}")
    print(f"H100 days to train all: {train_time(toks_per_batch)*num_total_batches/24}")


# pretrain ViT to classify all element 
# detect 100 fixed elements per screen
# output head will have shape 100 x [x,y,x+w,y+h, cls]
# this ensures that it can extract useful features for UI grounding 

# then, finetune this ViT + LLM
# project the pretrained ViT's latents into LLM
# LLM is trained on

# find one
# - input: help me find <x>
# - output: the <x> is at <bbox>

# understand ui
# - input: what should i do to <purpose> 
# - output: you should click(button)/read(text)/follow(link) the <ui> 
# at <bbox> which would allow <expectation> 

# list all 
# -input: list all ui elements
# -output: the ui elements in the screen are: <ui desc> at <bbox> ...

# ocr
# what's the text in box

# summarize
# purpose of screen - image encoder should be unfreezed for this

# not too concerned about reasoning
# about how to use buttons given high-lvl goal

# for now, deciding depth & patch size of vision encoder,
# and freeze encoder? 
# project wha

exit(-1)
'''
# Define a custom collate function
def collate_fn(batch):
    texts, images, labels = zip(*batch)
    return list(texts), list(images), list(labels)

ds = load_dataset("agentsea/wave-ui-25k")

dataloader = DataLoader(ds['train'], batch_size=BS, collate_fn=collate_fn, shuffle=True)

opt =  torch.optim.AdamW(v.parameters(), lr=3e-4)

for (x_texts, x_imgs),y in dataloader:
    toks = tokenizer.encode_batch(x_texts, padding=True)
    img_toks = 
'''