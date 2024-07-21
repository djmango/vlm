import torch
import time
import os
import sys
import re
import wandb
import torchvision.transforms as transforms
import deepspeed
import argparse

detr_path = os.path.join(os.path.dirname(__file__), 'detr')
sys.path.append(detr_path)

from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vit_pytorch.na_vit import NaViT
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from detr.models.detr import SetCriterion 
from detr.models.matcher import build_matcher
from data import apply_bbox_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.to_tensor = transforms.ToTensor()
        self.current_idx = 0
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self)
        item = self.dataset[idx]
        image = self.rescale_image(self.transform(item['image']), idx).to(device)
        
        content_boxes = [bbox for bbox in item['contentBoxes'] if bbox[2] >= bbox[0] and bbox[3] >= bbox[1]]
        key = item['key_name']
        label = item['labels']
        
        scale_w, scale_h = self.get_scale_factors(item['image'].size, key)
        scaled_boxes = [self.scale_bbox(box, scale_w, scale_h) for box in content_boxes]

        return image, scaled_boxes, label

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
        
        resized_img = transforms.Resize((target_h, target_w))(img)
        
        new_h = (target_h // self.patch_size) * self.patch_size
        new_w = (target_w // self.patch_size) * self.patch_size
        return resized_img[:, :new_h, :new_w].contiguous()

    def scale_bbox(self, bbox, scale_w, scale_h):
        return [
            bbox[0] * scale_w,
            bbox[1] * scale_h,
            bbox[2] * scale_w,
            bbox[3] * scale_h
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
            print(f"Batch creation time: {batch_time*1000:.4f} ms")

            if batch:
                yield batch
            else:
                break
        except IndexError:
            if batch:
                yield batch
            break

def display_img(img, bboxes, key):
    img_pil = transforms.ToPILImage()(img.cpu())
    
    for bbox in bboxes:
        img_pil = apply_bbox_to_image(img_pil, bbox)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_pil)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# model_engine, val_dataset, criterion, device, n_bboxs, n_classes
def validate(model_engine, val_dataset, criterion, device, n_bboxs, n_classes, BS, patch_size, max_batch_tokens):
    print('validating...')
    model_engine.eval()
    total_loss = 0
    total_cls_accuracy = 0
    total_iou_loss = 0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(get_batch(val_dataset, BS, patch_size, max_batch_tokens)):
            batched_imgs = []
            targets = []

            for sample in batch:
                imgs = []
                for img, bboxes, labels in sample:
                    target_dict = dict()
                    imgs.append(img.to(device))

                    sampled_idxs = torch.randperm(len(bboxes))[:min(n_bboxs, len(bboxes))]
                    
                    sampled_bboxes = torch.tensor(bboxes, device=device)[sampled_idxs]
                    sampled_labels = [labels[i][0] for i in sampled_idxs]
                    
                    bbox_target = torch.cat([sampled_bboxes, torch.zeros(n_bboxs - len(sampled_bboxes), 4, device=device)])
                    
                    target_dict['boxes'] = box_xyxy_to_cxcywh(bbox_target)
                    
                    labels = torch.tensor([cls_to_idx[label] for label in sampled_labels], dtype=torch.long, device=device)
                    
                    padded_labels = torch.full((n_bboxs,), n_classes, dtype=torch.long, device=device)
                    padded_labels[:len(labels)] = labels
                    
                    target_dict['labels'] = padded_labels
                    targets.append(target_dict)

                batched_imgs.append(imgs)

            out_cls, out_bbox = model_engine(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

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

cls_to_idx = {cls: idx for idx, cls in enumerate(sorted({
    'rowheader', 'doc-subtitle', 'application', 'table', 'contentinfo', 'columnheader', 'mark', 'menu', 'group', 'status', 
    'HeaderAsNonLandmark', 'link', 'DisclosureTriangle', 'PluginObject', 'Canvas', 'heading', 'menuitem', 'rowgroup', 
    'definition', 'document', 'option', 'DescriptionList', 'progressbar', 'Iframe', 'blockquote', 'toolbar', 'banner', 
    'list', 'emphasis', 'insertion', 'row', 'code', 'listitem', 'deletion', 'StaticText', 'Figcaption', 'LayoutTableCell', 
    'gridcell', 'LayoutTable', 'region', 'tablist', 'combobox', 'form', 'separator', 'Section', 'RootWebArea', 'superscript', 
    'slider', 'EmbeddedObject', 'button', 'Pre', 'article', 'LabelText', 'alert', 'tab', 'generic', 'IframePresentational', 
    'FooterAsNonLandmark', 'DescriptionListTerm', 'img', 'DescriptionListDetail', 'listbox', 'tabpanel', 'figure', 'Legend', 
    'radio', 'switch', 'log', 'navigation', 'paragraph', 'dialog', 'LineBreak', 'graphics-symbol', 'menubar', 'treeitem', 
    'note', 'LayoutTableRow', 'main', 'ListMarker', 'Ruby', 'complementary', 'subscript', 'Abbr', 'search', 'checkbox', 
    'textbox', 'time', 'strong'
}))}

assert len(cls_to_idx) == 88, f"Expected 88 classes, but got {len(cls_to_idx)}"

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default=None)
    parser.add_argument('--deepscale', action='store_true')
    parser.add_argument('--deepscale_config', type=str, default=None)
    # Add the following line to accept the local_rank argument
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def main():
    global BS, patch_size, max_batch_tokens
    args = parse_args() 

    deepspeed.init_distributed()


    # NaViT config
    logging = args.local_rank == 0
    BS = 6
    patch_size = 16
    max_img_size = 14*200 # 1920x1080
    max_batch_tokens = 1920//patch_size * 1080//patch_size
    n_classes = len(cls_to_idx)
    n_bboxs = 100
    dtype = torch.float32
    dim_head = 64
    n_heads = 2
    dim = 1024
    depth = 8
    # loss config
    EOS_CONF = 0.1
    CLS_WEIGHT = 1.0
    GIOU_WEIGHT = 2.0
    L1_WEIGHT = 5.0

    if logging:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="NaViT_training", config={
            "batch_size": 64,
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

    ds = load_dataset("biglab/webui-70k-elements", cache_dir="/workspace/cache")
    train_size = int(0.9 * len(ds['train']))
    train_dataset, val_dataset = torch.utils.data.random_split(ds['train'], [train_size, len(ds['train']) - train_size])
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

    epochs = 1000
    for epoch in range(epochs):

        model_engine.train()
        imgs_processed = 0
        epoch_time = time.time()

        for batch_idx, batch in enumerate(get_batch(train_dataset, BS, patch_size, max_batch_tokens)):
            if batch_idx > 0: break
            start_time = time.time()

            batched_imgs = []
            targets = []

            for sample in batch:
                imgs = []
                for img, bboxes, labels in sample:
                    target_dict = dict()
                    imgs.append(img.to(device))

                    sampled_idxs = torch.randperm(len(bboxes))[:min(n_bboxs, len(bboxes))]
                    
                    sampled_bboxes = torch.tensor(bboxes, device=device)[sampled_idxs]
                    sampled_labels = [labels[i][0] for i in sampled_idxs]
                    
                    bbox_target = torch.cat([sampled_bboxes, torch.zeros(n_bboxs - len(sampled_bboxes), 4, device=device)])
                    
                    target_dict['boxes'] = box_xyxy_to_cxcywh(bbox_target)
                    
                    labels = torch.tensor([cls_to_idx[label] for label in sampled_labels], dtype=torch.long, device=device)
                    
                    padded_labels = torch.full((n_bboxs,), n_classes, dtype=torch.long, device=device)
                    padded_labels[:len(labels)] = labels
                    
                    target_dict['labels'] = padded_labels
                    targets.append(target_dict)

                batched_imgs.append(imgs)

            out_cls, out_bbox = model_engine(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            model_engine.backward(losses)
            model_engine.step()

            imgs_processed += bs
            print(f'-- step {imgs_processed}/{len(train_dataset)} --')
            print(f"time so far: {time.time()-epoch_time:.2f} s")

            end_time = time.time()
            step_time = end_time - start_time

            for k in loss_dict:
                print(f'{k} loss: {loss_dict[k]:.3f}')
            print(f'total loss: {losses.item():.3f}')
            print(f"Step time: {step_time*1000:.4f} ms")

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
        '''
        val_loss, val_cls_accuracy, val_iou_loss = validate(
            model_engine, val_dataset, criterion, 
            device, n_bboxs, n_classes, BS,
            patch_size, max_batch_tokens
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
        model_engine.save_checkpoint(save_path)
        print(f"Model saved to {save_path}")
        '''
    if logging:
        wandb.finish()

if __name__ == '__main__':
    main()