import torch
import random
import time
import os
import re

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from vit_pytorch.na_vit import NaViT
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from data import apply_bbox_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BS = 2
patch_size = 14
max_img_size = 14*300 # 1920x1080
max_batch_tokens = 14400
n_bboxs = 100
dtype = torch.float32

v = NaViT(
    image_size = max_img_size,
    patch_size = patch_size,
    n_bboxs = n_bboxs,
    dim = 512,
    depth = 1,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1,
    token_dropout_prob = 0.1
).to(device)

#ds = load_dataset("biglab/webui-7k-elements")
ds = load_dataset("biglab/webui-7kbal-elements")

class NaViTDataset(Dataset):
    def __init__(self, dataset, patch_size, max_batch_size):
        self.dataset = dataset
        self.patch_size = patch_size
        self.max_batch_size = max_batch_size
        self.resolution_map = {
            r'default_(\d+)-(\d+)': lambda w, h: (int(w), int(h)),
            'iPad-Pro': (1138, 1518),
            'iPhone-13 Pro': (650, 1407)
        }

        self.current_idx = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self)
        print(idx)

        while True:
            item = self.dataset[idx]
            content_boxes = item['contentBoxes']
            # Check for invalid bounding boxes
            if self.has_valid_bboxes(content_boxes):
                break
            # If invalid, move to the next index
            idx = (idx + 1) % len(self)
        item = self.dataset[idx]
        image = self.rescale_image(transforms.ToTensor()(item['image']), idx).to(device)
        content_boxes = item['contentBoxes']
        key = item['key_name']
        
        # Rescale bounding boxes
        scale_w, scale_h = self.get_scale_factors(item['image'].size, key)
        scaled_boxes = [self.scale_bbox(box, scale_w, scale_h) for box in content_boxes]
        
        return image, scaled_boxes, key

    def get_target_resolution(self, key):
        for pattern, resolution in self.resolution_map.items():
            if isinstance(resolution, tuple):
                if key == pattern:
                    return resolution
            else:  # It's a lambda function
                match = re.match(pattern, key)
                if match:
                    return resolution(*match.groups())
        raise ValueError(f"Unknown key: {key}")

    def get_scale_factors(self, original_size, key):
        target_w, target_h = self.get_target_resolution(key)
        wildcard = 3 if key == 'iPhone-13 Pro' else (2 if key == 'iPad-Pro' else 1)
        return target_w / original_size[0] * wildcard, target_h / original_size[1] * wildcard

    def rescale_image(self, img, idx):
        c, h, w = img.shape
        key = self.dataset[idx]['key_name']
        target_w, target_h = self.get_target_resolution(key)
        
        # Resize the image
        resized_img = transforms.Resize((target_h, target_w))(img)
        
        # Crop to make divisible by patch_size
        new_h = (target_h // self.patch_size) * self.patch_size
        new_w = (target_w // self.patch_size) * self.patch_size
        return resized_img[:, :new_h, :new_w]

    def scale_bbox(self, bbox, scale_w, scale_h):
        return [
            bbox[0] * scale_w,
            bbox[1] * scale_h,
            bbox[2] * scale_w,
            bbox[3] * scale_h
        ]

    def has_valid_bboxes(self, bboxes):
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            if x0 > x1 or y0 > y1:
                return False
        return True

    def reset(self): self.current_idx = 0

def get_batch(dataset):
    dataset.reset()  # Reset the index at the start of each epoch
    while True:
        batch = []
        buffer = []
        current_toks = 0
        batch_start_time = time.time()  # Start timing the batch creation
        try:
            while len(batch) < BS:
                img, bbox, key = dataset[-1] # Always use index 0, the dataset handles progression
                img_tokens = (img.shape[1] // patch_size) * (img.shape[2] // patch_size)
                if current_toks + img_tokens > max_batch_tokens:
                    if buffer:
                        batch.append(buffer)
                        buffer = []
                        current_toks = 0
                buffer.append((img, bbox, key))
                current_toks += img_tokens

                if len(batch) == BS - 1 and buffer:
                    batch.append(buffer)
                    break

            batch_end_time = time.time()  # End timing the batch creation
            batch_time = batch_end_time - batch_start_time
            print(f"Batch creation time: {batch_time*1000:.4f} ms")

            if batch:
                yield batch
            else:
                break
        except IndexError:  # This will be raised when we've gone through the entire dataset
            if batch:
                yield batch
            break

def display_img(img, bboxes, key):
    img_pil = transforms.ToPILImage()(img.cpu())
    
    for bbox in bboxes:
        img_pil = apply_bbox_to_image(img_pil, bbox)
    
    fig, ax = plt.subplots(1)
    # Display the image with bounding boxes
    ax.imshow(img_pil)
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Show the plot
    plt.show()

# Create the dataset and dataloader
dataset = NaViTDataset(ds['train'], patch_size, max_batch_tokens)
#dataloader = DataLoader(navit_dataset, batch_size=BS, collate_fn=navit_collate_fn, shuffle=True)

def loss_fn(confidence_pred, bbox_pred, confidence_target, bbox_target):
    # Confidence loss (binary cross-entropy)
    confidence_loss = F.binary_cross_entropy_with_logits(confidence_pred, confidence_target)

    # bbox loss (only for positive samples)
    positive_mask = confidence_target > 0
    bbox_loss = F.mse_loss(bbox_pred[positive_mask], bbox_target[positive_mask])

    # Combine losses (you can adjust the weighting if needed)
    total_loss = confidence_loss + bbox_loss

    return total_loss

optim = torch.optim.Adam(v.parameters(), lr=3e-4)

for batch in get_batch(dataset):
    start_time = time.time()  # Start timing the step

    optim.zero_grad()  # Reset gradients at the start of each iteration

    batched_imgs = []
    confidence_targets = []
    bbox_targets = []

    for sample in batch:
        imgs = []
        for img, bboxes, _ in sample:
            imgs.append(img)

            sampled_bboxes = random.sample(bboxes, min(n_bboxs, len(bboxes)))
            
            confidence_target = [1] * len(sampled_bboxes) + [0] * (n_bboxs - len(sampled_bboxes))
            confidence_targets.append(confidence_target)
            
            bbox_target = sampled_bboxes + [[0,0,0,0]] * (n_bboxs - len(sampled_bboxes))
            bbox_targets.append(bbox_target)

        batched_imgs.append(imgs)

    confidence_target = torch.stack([torch.tensor(ct, device=device) for ct in confidence_targets]).to(dtype=dtype)
    bbox_target = torch.stack([torch.tensor(bt, device=device) for bt in bbox_targets]).to(dtype=dtype)

    print('target shapes:')
    print(confidence_target.shape)
    print(bbox_target.shape)

    preds = v(batched_imgs)
    preds = preds.view(confidence_target.shape[0], n_bboxs, 5)

    confidence_preds = preds[:, :, 0]  # Shape: (batch_size, 100)
    bboxs_preds = preds[:, :, 1:]     # Shape: (batch_size, 100, 4)

    print(confidence_preds.dtype)
    print(confidence_target.dtype)
    print(confidence_preds)
    print(bboxs_preds)

    loss = loss_fn(confidence_preds, bboxs_preds, confidence_target, bbox_target)
    loss.backward()
    optim.step()

    end_time = time.time()  # End timing the step
    step_time = end_time - start_time  # Calculate step time

    print(f"Loss: {loss.item()}")
    print(f"Step time: {step_time*1000:.4f} ms")

    '''
    for img, bbox, key in batch:
        print(img.shape)  # This should now print torch.Size([3, 714, 1274])
        display_img(img, bbox, key)
    '''