from vit_pytorch.na_vit import NaViT
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
import time
import os

BS = 64
patch_size = 8
v = NaViT(
    image_size = 256,
    patch_size = 8,
    num_classes=1000,
    dim = 1024,
    depth = 12,
    heads = 16,
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

def calc_memory(img_w, img_h, patch_size):
    num_patches = (img_w // patch_size) * (img_h // patch_size)
    print(f'Num patches (tokens): {num_patches}')
    patch_size_pixels = patch_size * patch_size * 3  # 3 channels for RGB
    # Memory for patches
    patch_memory = num_patches * patch_size_pixels * 4  # 4 bytes per float32
    # Memory for positional embeddings
    pos_embed_memory = num_patches * v.dim * 4
    # Memory for attention layers
    attention_memory = v.depth * (3 * v.dim * num_patches * 4)  # Q, K, V matrices
    # Memory for MLP layers
    mlp_memory = v.depth * (v.mlp_dim * num_patches * 4)
    # Total memory in bytes
    total_memory_bytes = patch_memory + pos_embed_memory + attention_memory + mlp_memory
    # Convert to GB
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    print(total_memory_gb)
    return total_memory_gb

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
    memory_usage = calc_memory(width, height, 12)
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