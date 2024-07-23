import torch
import time
import os
import sys
import wandb
import random
import deepspeed
import torchvision.transforms as transforms
import argparse
import numpy as np
from PIL import Image

detr_path = os.path.join(os.path.dirname(__file__), 'detr')
vit_path = os.path.join(os.path.dirname(__file__), 'vit-pytorch')
sys.path.append(detr_path)
sys.path.append(vit_path)

from torchvision.ops import box_iou
from vit_pytorch.na_vit import NaViT
from datasets import load_dataset
from torch.utils.data import Dataset
from detr.models.detr import SetCriterion 
from detr.models.matcher import build_matcher
from typing import List
from data import apply_bbox_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class COCODataset(Dataset):
    def __init__(self, dataset, patch_size, max_batch_size):
        self.dataset = dataset
        self.patch_size = patch_size
        self.max_batch_size = max_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.current_idx = 0
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % len(self)
            item = self.dataset[idx]
            objects = item['objects']
            boxes = objects['bbox']
            labels = objects['label']

            image = item['image'].convert('RGB')
            # Apply scale augmentation
            min_size = random.randint(480, 800)
            max_size = 1333
            image, scale_factor = self.resize_image(image, min_size, max_size)

            # Scale boxes according to image scaling
            boxes = [[box[0] * scale_factor, box[1] * scale_factor, 
                    box[2] * scale_factor, box[3] * scale_factor] for box in boxes]      
            # Apply random crop augmentation

            if random.random() < 0.5:
                image, boxes, labels = self.random_crop_with_boxes(image, boxes, labels)
                if len(boxes) > 0 and len(boxes) == len(labels):
                    break
                else:
                    print("invalid")
        
        # Ensure dimensions are divisible by patch_size
        w, h = image.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        image = image.resize((new_w, new_h))
        
        image = self.transform(image)

        # Convert from x,y,w,h to cx,cy,w,h format
        boxes = [[box[0] + box[2]/2, box[1] + box[3]/2, box[2], box[3]] for box in boxes]
        boxes = self.normalize_boxes(boxes, (new_w, new_h))
        return image, boxes, labels
    
    def random_crop_with_boxes(self, image, boxes, labels):
        w, h = image.size
        new_w = random.randint(int(0.5 * w), w)
        new_h = random.randint(int(0.5 * h), h)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        image = image.crop((left, top, left + new_w, top + new_h))
        
        # Adjust boxes
        adjusted_boxes = []
        adjusted_labels = []

        for box,label in zip(boxes, labels):
            x, y, bw, bh = box
            new_x = max(0, x - left)
            new_y = max(0, y - top)
            new_bw = min(new_w, x + bw - left) - new_x
            new_bh = min(new_h, y + bh - top) - new_y
            
            # Only keep boxes that are still within the cropped image
            if new_bw > 0 and new_bh > 0:
                adjusted_boxes.append([new_x, new_y, new_bw, new_bh])
                adjusted_labels.append(label)
        
        return image, adjusted_boxes, adjusted_labels

    def resize_image(self, image, min_size, max_size):
        w, h = image.size
        size = min(max(min_size, min(h, w)), max_size)
        scale_factor = size / min(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        if max(new_w, new_h) > max_size:
            scale_factor = max_size / max(w, h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        image = image.resize((new_w, new_h))
        return image, scale_factor      

    def random_crop(self, image):
        w, h = image.size
        new_w = random.randint(int(0.5 * w), w)
        new_h = random.randint(int(0.5 * h), h)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        return image.crop((left, top, left + new_w, top + new_h))

    def normalize_boxes(self, boxes, img_size):
        w, h = img_size
        normalized_boxes = []
        for box in boxes:
            cx, cy, bw, bh = box
            normalized_box = [
                max(0, min(1, cx / w)),
                max(0, min(1, cy / h)),
                max(0, min(1, bw / w)),
                max(0, min(1, bh / h))
            ]
            normalized_boxes.append(normalized_box)
        # Check for boxes that are out of range
        for i, box in enumerate(normalized_boxes):
            if not all(0 <= coord <= 1 for coord in box):
                print(f"Invalid box found at index {i}:")
                print(box)
                print("Corresponding image size:", img_size)
        
        assert all(all(0 <= coord <= 1 for coord in box) for box in normalized_boxes), "Normalized box coordinates must be between 0 and 1"
        return normalized_boxes

def display_img(img, bboxes, desc='sample_bbox', labels=None):
    img_pil = transforms.ToPILImage()(img.cpu())
    
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.view(-1, 4).tolist()
    
    if labels is None:
        labels = [None] * len(bboxes)

    for bbox, label in zip(bboxes, labels):
        img_pil = apply_bbox_to_image(img_pil, bbox, label=label)
    
    # Create the 'vis' directory if it doesn't exist
    os.makedirs('vis', exist_ok=True)
    
    # Save the image as a PNG file using PIL
    img_pil.save(f'vis/{desc}.png')

def get_batch(dataset, BS, patch_size, max_batch_tokens):
    while True:
        batch = []
        for _ in range(BS):
            idx = random.randint(0, len(dataset) - 1)
            item = dataset[idx]
            batch.append(item)
        yield batch

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default=None)
    parser.add_argument('--deepscale', action='store_true')
    parser.add_argument('--deepscale_config', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def unnormalize_coords(x, img_sizes):
    # x shape: [bs, n_bboxs, 4]
    # img_sizes shape: [bs, 2]
    assert x.dim() == img_sizes.dim()+1
    
    cx, cy, w, h = x.unbind(-1)
    imgw, imgh = img_sizes.unbind(-1)
    
    # Add dimensions to allow broadcasting
    if x.dim() == 3: imgw, imgh = imgw[:, None], imgh[:, None]  # [bs, 1, 1]
    
    # Unnormalize coordinates
    return torch.stack([cx * imgw, cy * imgh, w * imgw, h * imgh], dim=-1)


def main():
    global BS, patch_size, max_batch_tokens
    args = parse_args() 
    deepspeed.init_distributed()
    
    world_size = torch.distributed.get_world_size()

    logging = args.local_rank == 0 and 0
    BS = 2
    patch_size = 28
    max_img_size = patch_size * 200
    max_batch_tokens = 1333 // patch_size * 1333 // patch_size
    n_classes = 91  # COCO has 80 classes, but we add 1 for background
    n_bboxs = 100
    dim_head = 64
    n_heads = 8
    dim = 1024
    depth = 14
    epochs = 300//2  # As per DETR paper

    # loss config
    EOS_CONF = 0.1
    CLS_WEIGHT = 1.0
    GIOU_WEIGHT = 2.0
    L1_WEIGHT = 5.0

    if logging:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="NaViT_COCO", config={
            "epochs": epochs,
            "batch_size": BS,
            "patch_size": patch_size,
            "max_img_size": max_img_size,
            "max_batch_tokens": max_batch_tokens,
            "n_bboxs": n_bboxs,
            "dim_head": dim_head,
            "n_heads": n_heads,
            "dim": dim,
            "depth": depth
        })

    vit = NaViT(
        image_size = max_img_size,
        patch_size = patch_size,
        n_bboxs = n_bboxs,
        n_classes = n_classes,
        dim = dim,
        heads = n_heads,
        depth = depth,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        token_dropout_prob = 0.1
    ).to(device)

    #vit.init_weights()

    # Load COCO dataset
    train_dataset = load_dataset("rafaelpadilla/coco2017", split="train", cache_dir='/workspace/cache')
    val_dataset = load_dataset("rafaelpadilla/coco2017", split="val", cache_dir='/workspace/cache')

    # Split datasets based on world_size and local_rank
    world_size = torch.distributed.get_world_size()
    train_dataset = train_dataset.shard(num_shards=world_size, index=args.local_rank)
    val_dataset = val_dataset.shard(num_shards=world_size, index=args.local_rank)

    train_dataset = COCODataset(train_dataset, patch_size, max_batch_tokens)
    val_dataset = COCODataset(val_dataset, patch_size, max_batch_tokens)

    # Create DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=vit,
        model_parameters=vit.parameters()
    )

    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': CLS_WEIGHT, 'loss_bbox': L1_WEIGHT, 'loss_giou': GIOU_WEIGHT}
    matcher = build_matcher(cost_class=CLS_WEIGHT, cost_bbox=L1_WEIGHT, cost_giou=GIOU_WEIGHT)
    criterion = SetCriterion(n_classes, matcher, weight_dict, EOS_CONF, losses).to(device)

    if 1:
        view(train_dataset, BS, patch_size, max_batch_tokens, rank=args.local_rank)

    for epoch in range(epochs):

        model_engine.train()
        imgs_processed = 0

        for batch_idx, batch in enumerate(get_batch(train_dataset, BS, patch_size, max_batch_tokens)):
            start_time = time.time()

            batched_imgs = []
            targets = []
            img_sizes = []

            imgs = []
            for img, bboxes, labels in batch:
                target_dict = {}
                imgs.append(img.to(device))

                assert len(bboxes) == len(labels)

                num_to_sample = min(n_bboxs, len(bboxes))
                sampled_idxs = torch.randperm(len(bboxes))[:num_to_sample]
                
                sampled_bboxes = torch.tensor(bboxes)[sampled_idxs].to(device)
                sampled_labels = torch.tensor(labels)[sampled_idxs].to(device, dtype=torch.long)
                
                img_hw = [img.shape[0], img.shape[1]]
                
                target_dict['boxes'] = sampled_bboxes
                target_dict['labels'] = sampled_labels
                targets.append(target_dict)
                img_sizes.append(img_hw)

                batched_imgs.append(imgs)

            out_cls, out_bbox = model_engine(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_cls = torch.nn.functional.softmax(out_cls, dim=-1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            model_engine.backward(losses)
            model_engine.step()

            imgs_processed += BS

            end_time = time.time()
            step_time = end_time - start_time

            print(f'{losses.item():.4f}')

            if logging:
                log_dict = {
                    "total_loss": losses.item(),
                    "step_time_ms": step_time * 1000,
                    "batch": batch_idx,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                for k, v in loss_dict.items():
                    log_dict[f"{k}_loss"] = v.item()
                wandb.log(log_dict)

        print(f'Epoch {epoch} completed, {imgs_processed} images processed')

        # Validation
        val_loss = validate(model_engine, val_dataset, criterion, device, n_bboxs, n_classes, BS, patch_size, max_batch_tokens)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        if logging:
            wandb.log({
                "epoch": epoch+1,
                "val_loss": val_loss,
            })
        
        # Save model at the end of each epoch
        if (epoch + 1) % 50 == 0:
            save_path = f'vit_coco_checkpoint_epoch_{epoch+1}.pt'
            model_engine.save_checkpoint(save_path, epoch)
            print(f"Model saved to {save_path}")

        # Learning rate schedule
        if epoch == 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    if logging:
        wandb.finish()

def validate(model_engine, val_dataset, criterion, device, n_bboxs, n_classes, BS, patch_size, max_batch_tokens):
    model_engine.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in get_batch(val_dataset, BS, patch_size, max_batch_tokens):
            images, boxes, labels = zip(*batch)
            images = torch.stack(images).to(device)
            
            # Convert boxes to cxcywh format
            boxes = [box_xyxy_to_cxcywh(b) for b in boxes]
            
            targets = [{'boxes': b.to(device), 'labels': l.to(device)} for b, l in zip(boxes, labels)]

            out_cls, out_bbox = model_engine(images)
            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            total_loss += losses.item()
            num_batches += 1

            if num_batches >= 50:  # Validate on 50 batches to save time
                break

    avg_loss = total_loss / num_batches
    model_engine.train()
    return avg_loss

def view(dataset, BS, patch_size, max_batch_tokens, n_batches=10, rank=0):
    batch_generator = get_batch(dataset, BS, patch_size, max_batch_tokens)
    
    for batch_idx in range(n_batches):
        batch = next(batch_generator)
        
        for img_idx, (img, bboxes, labels) in enumerate(batch):
            # Convert bboxes from cxcywh to xyxy format
            bboxes_xyxy = box_cxcywh_to_xyxy(torch.tensor(bboxes))
            
            # Denormalize bboxes
            img_size = torch.tensor([img.shape[2], img.shape[1]])  # width, height
            bboxes_denorm = bboxes_xyxy * img_size.repeat(2)
            
            # Display and save the image
            display_img(img, bboxes_denorm, desc=f'rank_{rank}_batch_{batch_idx}_img_{img_idx}', labels=labels)
    
    print(f"Saved {n_batches * BS} images with bounding boxes in the 'vis' directory.")

# Add this to the main function or where appropriate
if __name__ == '__main__':
    main()