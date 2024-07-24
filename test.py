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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from detr.models.detr import SetCriterion 
from detr.models.matcher import _build_matcher
from detr.datasets import build_dataset, get_coco_api_from_dataset
from detr.datasets.coco_eval import CocoEvaluator
from detr.util.misc import collate_fn
from typing import List
from data import apply_bbox_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

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

    logging = args.local_rank == 0 and 1
    BS = 2
    patch_size = 32
    max_img_size = patch_size * 200
    max_batch_tokens = 1333 // patch_size * 1333 // patch_size
    # https://gist.githubusercontent.com/AruniRC/7b3dadd004da04c80198557db5da4bda/raw/2f10965ace1e36c4a9dca76ead19b744f5eb7e88/ms_coco_classnames.txt
    n_classes = 81  # COCO has 80 classes, but we add 1 for background 
    n_bboxs = 100
    dim_head = 64
    n_heads = 6
    dim = 1024
    head_dim = int(dim * 2)
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
        head_dim = head_dim,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        token_dropout_prob = 0.1
    ).to(device)

    #vit.init_weights()
    postprocessors = {'bbox': PostProcess()}

    # Load COCO dataset
    base_ds = get_coco_api_from_dataset(dataset_val)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    world_size = torch.distributed.get_world_size()

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=world_size)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
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
    criterion = SetCriterion(n_classes, matcher, weight_dict, EOS_CONF, losses).to(device)

    for epoch in range(epochs):

        model_engine.train()
        imgs_processed = 0

        for i, samples, targets in enumerate(data_loader_train):
            if i > 0: break
            # both are torch.tensor
            start_time = time.time()
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            batched_imgs = [samples[i].to(device) for i in range(len(samples))]
            out_cls, out_bbox = model_engine(batched_imgs)

            bs, _ = out_cls.shape
            out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
            out_bbox = out_bbox.view(bs, n_bboxs, 4)

            outputs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            model_engine.backward(losses)
            model_engine.step()

            imgs_processed += BS

            end_time = time.time()
            step_time = end_time - start_time

            print(f'{losses.item():.4f} L, {step_time*1000:.4f} ms, {imgs_processed} imgs')

            if logging:
                log_dict = {
                    "total_loss": losses.item(),
                    "step_time_ms": step_time * 1000,
                    "batch": i,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                for k, v in loss_dict.items():
                    log_dict[f"{k}_loss"] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(log_dict)

        print(f'Epoch {epoch} completed, {imgs_processed} images processed')

        # Validation
        val_loss = validate(
            model_engine, data_loader_val, criterion, 
            base_ds, postprocessors, n_bboxs, n_classes
            )
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        if logging:
            wandb.log({
                "epoch": epoch+1,
                "val_loss": val_loss,
            })
        
        # Save model at the end of each epoch
        save_path = f'vit_coco_checkpoint_epoch_{epoch+1}.pt'
        model_engine.save_checkpoint(save_path, epoch)
        print(f"Model saved to {save_path}")


    if logging:
        wandb.finish()

@torch.no_grad()
def validate(model_engine, val_dataset, criterion, base_ds, postprocessors, n_bboxs, n_classes):
    model_engine.eval()
    criterion.eval()

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for i, samples, targets in enumerate(val_dataset):
        # both are torch.tensor
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        batched_imgs = [samples[i].to(device) for i in range(len(samples))]
        out_cls, out_bbox = model_engine(batched_imgs)

        bs, _ = out_cls.shape
        out_cls = out_cls.view(bs, n_bboxs, n_classes+1)
        out_bbox = out_bbox.view(bs, n_bboxs, 4)

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
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

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