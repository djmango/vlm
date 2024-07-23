import torch
import time
import os
import sys
import re
import wandb
import random
import deepspeed
import torchvision.transforms as transforms
import argparse
import numpy as np

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
from data import apply_bbox_to_image
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def validbox(box, bounds):
    left,bottom,right,top = bounds
    x0,y0,x1,y1 = box
    if (x0 >= left) and (y0 >= bottom) and (x1 >= x0) and (y1 >= y0) and (x1 <= right) and (y1 <= top):
        if (x1 - x0 >= 15) and (y1 - y0 >= 15):
            return 1
    return 0

def validlabel(labels: List[str]):
    if len(labels) > 2:
        return False
    return not any(label in blacklist for label in labels)
    

class RandomCropWithBoxes(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes):
        h, w = image.shape[1:]
        new_h, new_w = self.size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[:, top:top+new_h, left:left+new_w]
        boxes = [[
            max(0, box[0] - left),
            max(0, box[1] - top),
            min(new_w, box[2] - left),
            min(new_h, box[3] - top)
        ] for box in boxes]

        return image, boxes

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
        def ensure_rgb(image):
            if image.shape[0] == 4:  # If the image has 4 channels (RGBA)
                return image[:3, :, :]  # Return only the first 3 channels (RGB)
            return image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(ensure_rgb),
        ])
        self.to_tensor = transforms.ToTensor()
        self.current_idx = 0
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # skip iPhone-13 Pro
        while True:
            idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % len(self)
            item = self.dataset[idx]
            if item['key_name'] != 'iPhone-13 Pro':
                break
            
        image = self.transform(item['image'])
        key = item['key_name']
        labels = item['labels']
        bboxs = item['contentBoxes']  # Add this line to define bboxs

        scale_w, scale_h = self.get_scale_factors(item['image'].size, key)
        scaled_boxes = [self.scale_bbox(box, scale_w, scale_h) for box in bboxs]
        image, bboxs, labels = self.resize_and_augment(image, key, scaled_boxes, labels)

        return image, bboxs, labels

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
        wildcard = 2 if key == 'iPad-Pro' else 1
        return target_w / original_size[0] * wildcard, target_h / original_size[1] * wildcard

    def resize_and_augment(self, img, key, bboxs, labels):

        target_w, target_h = self.get_target_resolution(key)
        img = transforms.Resize((target_h, target_w))(img)
        augment = random.random() < 0.42

        if augment:
            min_crop_size = (img.shape[1] // 2, img.shape[2] // 2)
            max_crop_size = (img.shape[1]-1, img.shape[2]-1)

            random_crop = RandomCropWithBoxes((
                random.randint(min_crop_size[0], max_crop_size[0]),
                random.randint(min_crop_size[1], max_crop_size[1])
            ))
            img, bboxs = random_crop(img, bboxs)

        # Ensure dimensions are divisible by patch_size
        new_h = (img.shape[1] // self.patch_size) * self.patch_size
        new_w = (img.shape[2] // self.patch_size) * self.patch_size

        # Crop to ensure dimensions are divisible by patch_size
        final_img = img[:, :new_h, :new_w].contiguous()
        valid_labels = []
        valid_boxes = []

        for bbox, label in zip(bboxs, labels):
            if validbox(bbox, (0,0,new_w,new_h)) and validlabel(label):
                if len(label) > 1 and 'StaticText' in label:
                    label = [l for l in label if l != 'StaticText']
                valid_labels.append(label)
                valid_boxes.append(bbox)
        return final_img, valid_boxes, valid_labels

    def scale_bbox(self, bbox, scale_w, scale_h):
        return [
            bbox[0] * scale_w, bbox[1] * scale_h,
            bbox[2] * scale_w,bbox[3] * scale_h
        ]

    def reset(self): self.current_idx = 0

def get_batch(dataset, BS, patch_size, max_batch_tokens):
    dataset.reset()
    while True:
        batch = []
        buffer = []
        current_toks = 0
        batch_start_time = time.time()
        try:
            while len(batch) < BS:
                img, bbox, label = dataset[-1]
                img_tokens = (img.shape[1] // patch_size) * (img.shape[2] // patch_size)
                if current_toks + img_tokens > max_batch_tokens:
                    if buffer:
                        batch.append(buffer)
                        buffer = []
                        current_toks = 0
                buffer.append((img, bbox, label))
                current_toks += img_tokens

                if len(batch) == BS - 1 and buffer:
                    batch.append(buffer)
                    break

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            #print(f"Batch creation time: {batch_time*1000:.4f} ms")

            if batch:
                yield batch
            else:
                break
        except IndexError:
            if batch:
                yield batch
            break

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


blacklist = {
    'separator', 'LayoutTableCell', 'LayoutTableRow', 'generic', 'Pre', 'LineBreak',
    'group', 'application', 'document', 'emphasis', 'strong', 'insertion', 'deletion',
    'superscript', 'subscript', 'LayoutTable', 'RootWebArea', 'ListMarker', "spinbutton", 
    'spinbutton'
}

cls_to_idx = {cls: idx for idx, cls in enumerate(sorted({
    cls for cls in {
    'rowheader', 'doc-subtitle', 'table', 'contentinfo', 'columnheader', 'mark', 'menu', 'status', 
    'HeaderAsNonLandmark', 'link', 'DisclosureTriangle', 'PluginObject', 'Canvas', 'heading', 'menuitem', 'rowgroup', 
    'definition', 'option', 'DescriptionList', 'progressbar', 'Iframe', 'blockquote', 'toolbar', 'banner', 
    'list', 'row', 'code', 'listitem', 'StaticText', 'Figcaption', 
    'gridcell', 'region', 'tablist', 'combobox', 'form', 'Section', 
    'slider', 'EmbeddedObject', 'button', 'article', 'LabelText', 'alert', 'tab', 'IframePresentational', 
    'FooterAsNonLandmark', 'DescriptionListTerm', 'img', 'DescriptionListDetail', 'listbox', 'tabpanel', 'figure', 'Legend', 
    'radio', 'switch', 'log', 'navigation', 'paragraph', 'dialog', 'graphics-symbol', 'menubar', 'treeitem', 
    'note', 'main', 'Ruby', 'complementary', 'Abbr', 'search', 'checkbox', 
    'textbox', 'time', 'caption', 
    } if cls not in blacklist
}))}

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1).to(dtype=x.dtype)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1).to(dtype=x.dtype)

def normalize_coords(x, img_sizes):
    # x shape: [bs, n_bboxs, 4]
    # img_sizes shape: [bs, 2]
    assert x.dim() == img_sizes.dim()+1
    
    x0, y0, x1, y1 = x.unbind(-1)
    imgh, imgw = img_sizes.unbind(-1)
    
    # Add dimensions to allow broadcasting
    if x.dim() == 3: imgw, imgh = imgw[:, None], imgh[:, None]  # [bs, 1, 1]

    # Normalize coordinates
    ret = torch.stack([x0 / imgw, y0 / imgh, x1 / imgw, y1 / imgh], dim=-1)
    
    # Assert that all values in ret are between 0 and 1 (inclusive)
    assert torch.all((ret >= 0) & (ret <= 1)), f"Normalized coordinates must be between 0 and 1, at {img_sizes}, {x}"
    
    return ret

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

# model, val_dataset, criterion, device, n_bboxs, n_classes
def validate(model_engine, val_dataset, criterion, device, n_bboxs, n_classes, BS, patch_size, max_batch_tokens, epoch=0):
    print('validating...')
    model_engine.eval()
    total_loss = 0
    total_cls_accuracy = 0
    total_iou_loss = 0
    num_samples = 0
    i = 0
    j = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(get_batch(val_dataset, BS, patch_size, max_batch_tokens)):
            if batch_idx > 0: break
            batched_imgs = []
            targets = []
            img_sizes = []
            for sample in batch:
                imgs = []
                for img, bboxes, labels in sample:
                    target_dict = dict()
                    imgs.append(img.to(device))

                    sampled_idxs = torch.randperm(len(bboxes))[:min(n_bboxs, len(bboxes))]
                    
                    sampled_bboxes = torch.tensor(bboxes, device=device)[sampled_idxs]
                    sampled_labels = [labels[i][0] for i in sampled_idxs]
                    
                    bbox_target = torch.cat([sampled_bboxes, torch.zeros(n_bboxs - len(sampled_bboxes), 4, device=device)])
                    bbox_target = box_xyxy_to_cxcywh(bbox_target)
                    img_hw = [img.shape[1], img.shape[2]]
                    normalized_bbox_target = normalize_coords(bbox_target, torch.tensor(img_hw))

                    display_img(img, sampled_bboxes, desc=f"{epoch}_val_target_{i}")
                    i+=1
                    
                    labels = torch.tensor([cls_to_idx[label] for label in sampled_labels], dtype=torch.long, device=device)
                    padded_labels = torch.full((n_bboxs,), n_classes, dtype=torch.long, device=device)
                    padded_labels[:len(labels)] = labels
                    
                    target_dict['boxes'] = normalized_bbox_target
                    target_dict['labels'] = padded_labels
                    targets.append(target_dict)
                    img_sizes.append(img_hw)

                batched_imgs.append(imgs)

            out_cls, out_bbox = model_engine(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            pred_bboxes = box_cxcywh_to_xyxy(unnormalize_coords(out_bbox, torch.tensor(img_sizes, device=device)))

            ii = 0
            for sample in batch:
                for img, _, _ in sample:
                    #print(targets[i]['boxes'])
                    #print(out_bbox[i])
                    display_img(img, pred_bboxes[ii], desc=f"{epoch}_val_pred_{j}")
                    ii += 1
                    j += 1

            loss_dict = criterion(outs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            pred_cls = out_cls.argmax(dim=-1)
            target_cls = torch.stack([t['labels'] for t in targets])
            cls_accuracy = (pred_cls == target_cls).float().mean()

            pred_boxes = box_cxcywh_to_xyxy(out_bbox)
            target_boxes = torch.stack([box_cxcywh_to_xyxy(t['boxes']) for t in targets])
            
            indices = criterion.matcher(outs, targets)
            
            iou_loss = 0
            for idx, (pred_idx, target_idx) in enumerate(indices):
                if len(pred_idx) > 0:
                    matched_pred_boxes = pred_boxes[idx][pred_idx]
                    matched_target_boxes = target_boxes[idx][target_idx]
                    iou = box_iou(matched_pred_boxes, matched_target_boxes)
                    iou_loss += (1 - iou.diag()).mean()
            
            total_loss += losses.item()
            total_cls_accuracy += cls_accuracy.item()
            total_iou_loss += iou_loss
            num_samples += bs

    avg_loss = total_loss / num_samples
    avg_cls_accuracy = total_cls_accuracy / num_samples
    avg_iou_loss = total_iou_loss / num_samples
    model_engine.train()
    return avg_loss, avg_cls_accuracy, avg_iou_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default=None)
    parser.add_argument('--deepscale', action='store_true')
    parser.add_argument('--deepscale_config', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def main():
    global BS, patch_size, max_batch_tokens
    args = parse_args() 
    new_labels = set()
    deepspeed.init_distributed()
    logging = args.local_rank == 0 and 1
    BS = 4
    patch_size = 28
    max_img_size = patch_size*200 # 1920x1080
    max_batch_tokens = 1920//patch_size * 1080//patch_size
    n_classes = len(cls_to_idx)
    n_bboxs = 100
    dim_head = 64
    n_heads = 4
    dim = 1024
    depth = 14
    epochs = 30
    # loss config
    EOS_CONF = 0.1
    CLS_WEIGHT = 1.0
    GIOU_WEIGHT = 2.0
    L1_WEIGHT = 5.0

    if logging:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="NaViT_training", config={
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

    # Call this method after model initialization
    vit.init_weights()

    ds = load_dataset("biglab/webui-70k-elements")
    train_size = int(0.9 * len(ds['train']))
    train_dataset, val_dataset = torch.utils.data.random_split(
        ds['train'], 
        [train_size, len(ds['train']) - train_size],
    )
    train_dataset = NaViTDataset(train_dataset, patch_size, max_batch_tokens)
    val_dataset = NaViTDataset(val_dataset, patch_size, max_batch_tokens)

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

    for epoch in range(epochs):
        model_engine.train()
        imgs_processed = 0

        for batch_idx, batch in enumerate(get_batch(train_dataset, BS, patch_size, max_batch_tokens)):
            start_time = time.time()

            batched_imgs = []
            targets = []
            img_sizes = []

            for sample in batch:
                imgs = []
                for img, bboxes, labels in sample:
                    target_dict = dict()

                    # change back to randperm later
                    sampled_idxs = torch.arange(min(n_bboxs, len(bboxes)))
                    sampled_bboxes = torch.tensor(bboxes, device=device)[sampled_idxs]
                    sampled_labels = [labels[i][0] for i in range(min(n_bboxs, len(bboxes)))]
                    
                    #print(f'number of bboxes: {len(sampled_bboxes)}')
                    bbox_target = torch.cat([sampled_bboxes, torch.zeros(n_bboxs - len(sampled_bboxes), 4, device=device)])
                    bbox_target = box_xyxy_to_cxcywh(bbox_target)
                    img_hw = [img.shape[1], img.shape[2]]
                    normalized_bbox_target = normalize_coords(bbox_target, torch.tensor(img_hw))

                    try:
                        labels = torch.tensor([cls_to_idx[label] for label in sampled_labels], dtype=torch.long, device=device)
                        padded_labels = torch.full((n_bboxs,), n_classes, dtype=torch.long, device=device)
                        padded_labels[:len(labels)] = labels
                        
                        target_dict['boxes'] = normalized_bbox_target
                        target_dict['labels'] = padded_labels

                        imgs.append(img.to(device))
                        targets.append(target_dict)
                        img_sizes.append(img_hw)
                        imgs_processed += 1
                    except KeyError:
                        for label in sampled_labels:
                            if label not in cls_to_idx:
                                new_labels.add(label)
                        if new_labels:
                            print(f"Warning: Found new labels not in cls_to_idx: {new_labels}")
                        continue
                
                if imgs:
                    batched_imgs.append(imgs)

            out_cls, out_bbox = vit(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            '''
            pred_bboxes = box_cxcywh_to_xyxy(unnormalize_coords(out_bbox, torch.tensor(img_sizes, device=device)))
            i = 0
            for sample in batch:
                for img, _, _ in sample:
                    display_img(img, pred_bboxes[i], desc=f"rank={args.local_rank}_{i}")
                    i += 1
            '''

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            model_engine.backward(losses)
            model_engine.step()

            imgs_processed += bs

            end_time = time.time()
            step_time = end_time - start_time

            print(f'{losses.item():.4f}\n')

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

        print(f'epoch {epoch}\n')
        print(f'{imgs_processed} imgs processed rank {args.local_rank}\n')

        val_loss, val_cls_accuracy, val_iou_loss = validate(
            vit, val_dataset, criterion, 
            device, n_bboxs, n_classes, BS,
            patch_size, max_batch_tokens,
            epoch=epoch
        )

        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, "
            f"Classification Accuracy: {val_cls_accuracy:.4f}, IoU Loss: {val_iou_loss:.4f}")

        if logging:
            wandb.log({
                "epoch": epoch+1,
                "val_loss": val_loss,
                "val_cls_accuracy": val_cls_accuracy,
                "val_iou_loss": val_iou_loss
            })
        
        # Save model at the end of each epoch
        save_path = f'vit_checkpoint_epoch_{epoch+1}.pt'
        model_engine.save_checkpoint(save_path, epoch)
        print(f"Model saved to {save_path}")

        # Save new_labels to new_labels.txt
        if new_labels:
            with open('new_labels.txt', 'a') as f:
                for label in sorted(new_labels):
                    f.write(f"{label}\n")
            new_labels = set()
            print(f"New labels saved to new_labels.txt")

    if logging:
        wandb.finish()



if __name__ == '__main__':
    main()
