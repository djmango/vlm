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
from vit_pytorch.det_vit import ViT
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from detr.models.detr import SetCriterion 
from detr.models.matcher import _build_matcher
from detr.datasets import build_dataset, get_coco_api_from_dataset
from detr.datasets.coco_eval import CocoEvaluator
from detr.util.misc import collate_fn
from typing import List
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.distributed import all_reduce, ReduceOp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(samples, targets, patch_size=32):
    # Function to round up to the nearest multiple of patch_size
    def round_up(x, p):
        return ((x + p - 1) // p) * p

    processed_samples = []
    processed_targets = []

    for img, target in zip(samples.tensors, targets):
        # Get original dimensions
        c, h, w = img.shape
        
        # Calculate new dimensions
        new_h = round_up(h, patch_size)
        new_w = round_up(w, patch_size)
        
        # Resize image
        resized_img = T.resize(img, (new_h, new_w), antialias=True)
        
        # Adjust bounding boxes
        if 'boxes' in target:
            boxes = target['boxes']
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)
            target['boxes'] = boxes

        processed_samples.append(resized_img)
        processed_targets.append(target)

    # Stack processed samples
    processed_samples = torch.stack(processed_samples)

    return processed_samples, processed_targets

class PostProcess(torch.nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

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
    # x shape: [bs, num_bboxs, 4]
    # img_sizes shape: [bs, 2]
    assert x.dim() == img_sizes.dim()+1
    
    cx, cy, w, h = x.unbind(-1)
    imgw, imgh = img_sizes.unbind(-1)
    
    # Add dimensions to allow broadcasting
    if x.dim() == 3: imgw, imgh = imgw[:, None], imgh[:, None]  # [bs, 1, 1]
    
    # Unnormalize coordinates
    return torch.stack([cx * imgw, cy * imgh, w * imgw, h * imgh], dim=-1)

def load_eva_ckpt(path, vit, keys_to_del=[]):
    checkpoint = torch.load(path, map_location='cuda')
    
    for key in list(checkpoint.keys()):
        if any(k in key for k in keys_to_del):
            del checkpoint[key]

    vit.load_state_dict(checkpoint, strict=False)

    n_parameters = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"Loaded model with {n_parameters:,} parameters")

#deepspeed --num_gpus=4 vit_coco.py --deepspeed --deepspeed_config ds_config.json
def main():

    global BS, patch_size, max_batch_tokens
    args = parse_args() 
    deepspeed.init_distributed()

    world_size = torch.distributed.get_world_size()
    logging = args.local_rank == 0 and 1
    BS = 2
    patch_size = 16
    max_img_size = 1440
    # https://gist.githubusercontent.com/AruniRC/7b3dadd004da04c80198557db5da4bda/raw/2f10965ace1e36c4a9dca76ead19b744f5eb7e88/ms_coco_classnames.txt
    num_classes = 91  # COCO has 80 classes, but we add 1 for background 
    num_bboxs = 100
    dim_head = 64
    num_heads = 8
    dim = 1024
    class_head_dim = int(dim * 2)
    depth = 12
    epochs = 300  # As per DETR paper
    dtype = torch.float16

    eva_path = '/workspace/vlm/eva_coco_2_checkpoint_epoch_6.pt/eva_coco.pt'

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
            "num_bboxs": num_bboxs,
            "dim_head": dim_head,
            "num_heads": num_heads,
            "dim": dim,
            "depth": depth
        })

    vit = ViT(
        image_size = max_img_size,
        patch_size = patch_size,
        num_bboxs = num_bboxs,
        num_classes = num_classes,
        dim = dim,
        heads = num_heads,
        depth = depth,
        dim_head = dim_head,
        mlp_dim = 2048,
        class_head_dim = int(dim * 2)
    )

    load_eva_ckpt(eva_path, vit)

    vit = vit.to(device, dtype=dtype)

    num_parameters = sum(p.numel() for p in vit.parameters() if p.requires_grad)

    postprocessors = {'bbox': PostProcess()}

    class Args:
        def __init__(self):
            self.dataset_file = 'coco'
            self.masks = False
            self.coco_path = os.getenv('COCO_PATH')
    
    data_args = Args()

    dataset_train = build_dataset(image_set='train', args=data_args)
    dataset_val = build_dataset(image_set='val', args=data_args)

    base_ds = get_coco_api_from_dataset(dataset_val)

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, BS, drop_last=True)
    
    world_size = torch.distributed.get_world_size()

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=world_size)
    data_loader_val = DataLoader(dataset_val, BS, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=world_size)

    # Split datasets based on world_size and local_rank
    # Create DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=vit,
        model_parameters=vit.parameters()
    )

    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': CLS_WEIGHT, 'loss_bbox': L1_WEIGHT, 'loss_giou': GIOU_WEIGHT}
    matcher = _build_matcher(cost_class=CLS_WEIGHT, cost_bbox=L1_WEIGHT, cost_giou=GIOU_WEIGHT)
    criterion = SetCriterion(num_classes, matcher, weight_dict, EOS_CONF, losses).to(device, dtype=dtype)

    for epoch in range(epochs):

        model_engine.train()
        imgs_processed = 0

        print("this many imgs per epoch: ")
        print(len(data_loader_train)*BS)

        for i, (samples, targets, _) in enumerate(data_loader_train):

            samples, targets = preprocess(samples, targets, patch_size=patch_size)

            # both are torch.tensor
            start_time = time.time()
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            out_cls, out_bbox = model_engine(samples.to(device, dtype=dtype))

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, num_bboxs, num_classes+1)
            out_bbox = out_bbox.view(bs, num_bboxs, 4)

            outputs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            weighted_loss_dict = {k: v * weight_dict[k] if k in weight_dict else v for k, v in loss_dict.items()}
            losses = sum(weighted_loss_dict.values())

            model_engine.backward(losses)
            model_engine.step()

            imgs_processed += BS
            
            # Perform all_reduce for each loss component
            all_reduce_start_time = time.time()
            
            loss_ce = weighted_loss_dict.get('loss_ce', torch.tensor(0.0).to(device))
            loss_bbox = weighted_loss_dict.get('loss_bbox', torch.tensor(0.0).to(device))
            loss_giou = weighted_loss_dict.get('loss_giou', torch.tensor(0.0).to(device))

            all_reduce(loss_ce, op=ReduceOp.SUM)
            all_reduce(loss_bbox, op=ReduceOp.SUM)
            all_reduce(loss_giou, op=ReduceOp.SUM)

            # Calculate mean values
            mean_loss_ce = (loss_ce / world_size).item()
            mean_loss_bbox = (loss_bbox / world_size).item()
            mean_loss_giou = (loss_giou / world_size).item()

            # Create reduced_loss_dict with averaged values
            reduced_loss_dict = {
                'loss_ce': mean_loss_ce,
                'loss_bbox': mean_loss_bbox,
                'loss_giou': mean_loss_giou
            }

            # Calculate total loss from reduced_loss_dict
            total_loss = sum(reduced_loss_dict.values())

            end_time = time.time()
            step_time = end_time - start_time
            all_reduce_time = end_time - all_reduce_start_time

            print(f"All reduce time: {all_reduce_time*1000:.4f} ms")
            print(f'{total_loss:.4f} L, {step_time*1000:.4f} ms, {imgs_processed*world_size} imgs')

            if logging:
                log_dict = {
                    "total_loss": total_loss,
                    "step_time_ms": step_time * 1000,
                    "batch": i,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                log_dict.update({f"{k}_loss": v for k, v in reduced_loss_dict.items()})
                wandb.log(log_dict)

        print(f'Epoch {epoch}/{epochs} completed, {imgs_processed} images processed')

        # Validation
        stats, _ = validate(
            model_engine, data_loader_val, criterion, 
            base_ds, postprocessors, num_bboxs, num_classes
            )

        if logging:
            log_stats = {**{f'test_{k}': v for k, v in stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            print(log_stats)
            wandb.log(log_stats)
        
        # Save model at the end of each epoch
        save_path = f'det_vit_coco_checkpoint_epoch_{epoch}.pt'
        model_engine.save_checkpoint(save_path, epoch)
        print(f"Model saved to {save_path}")

    if logging:
        wandb.finish()

@torch.no_grad()
def validate(model_engine, val_dataset, criterion, base_ds, postprocessors, num_bboxs, num_classes):
    model_engine.eval()
    criterion.eval()

    iou_types = tuple(k for k in ('bbox') if k in postprocessors.keys())

        # Fix: Convert iou_types to a list if it's empty
    if not iou_types:
        iou_types = ['bbox']

    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for i, (samples, targets) in enumerate(val_dataset):

        samples, targets = preprocess(samples, targets, patch_size=patch_size)
        # both are torch.tensor
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        out_cls, out_bbox = model_engine(samples.to(device, dtype=dtype))

        bs, _ = out_cls.shape
        out_cls = out_cls.view(bs, num_bboxs, num_classes+1)
        out_bbox = out_bbox.view(bs, num_bboxs, 4)

        outputs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)        

    stats = dict()
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats

    return stats, coco_evaluator

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