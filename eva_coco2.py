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

eva_path = os.path.join(os.path.dirname(__file__), 'EVA/EVA-CLIP/rei')
detr_path = os.path.join(os.path.dirname(__file__), 'detr')
vit_path = os.path.join(os.path.dirname(__file__), 'vit-pytorch')

sys.path.append(eva_path)
sys.path.append(detr_path)
sys.path.append(vit_path)

from torchvision.ops import box_iou
from vit_pytorch.eva import ViT
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from detr.models.detr import SetCriterion 
from detr.models.matcher import _build_matcher
from detr.datasets import build_dataset, get_coco_api_from_dataset
from detr.util.misc import collate_fn
from typing import List
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
#from torchvision.transforms import Compose

from torch.distributed import all_reduce, ReduceOp
from eva_clip import create_model_and_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_dtype(torch.float16)
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def build_teacher_transform():
    image_size = 336
    
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
    ])

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
        resized_img = torchvision.transforms.functional.resize(img, (new_h, new_w), antialias=True)
        
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default=None)
    parser.add_argument('--deepscale', action='store_true')
    parser.add_argument('--deepscale_config', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def compute_loss(output, target):
    loss_func = torch.nn.CosineSimilarity(dim=-1)
    loss = loss_func(output.float(), target.float())
    return -loss.mean()

# deepspeed --num_gpus=4 eva_coco2.py --deepspeed --deepspeed_config ds_config.json
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
    n_bboxs = 100
    dim_head = 64
    n_heads = 8
    dim = 1024
    class_head_dim = int(dim * 2)
    depth = 12
    epochs = 300  # As per DETR paper
    dtype = torch.float16

    # loss config
    EOS_CONF = 0.1
    CLS_WEIGHT = 1.0
    GIOU_WEIGHT = 2.0
    L1_WEIGHT = 5.0

    if logging:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="EVA_COCO", config={
            "epochs": epochs,
            "batch_size": BS,
            "patch_size": patch_size,
            "max_img_size": max_img_size,
            "dim_head": dim_head,
            "n_heads": n_heads,
            "dim": dim,
            "depth": depth
        })

    # 4.4B parameter 
    # around 8.8GB in RAM
    #model_name = "EVA02-CLIP-bigE-14-plus" 
    #pretrained = "/workspace/vlm/EVA02_CLIP_E_psz14_plus_s9B.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

    # 1B parameter 
    model_name = 'EVA02-CLIP-L-14-336'
    pretrained = '/workspace/vlm/EVA02_CLIP_L_336_psz14_s6B.pt'
    # model+

    # load teacher
    teacher, _, p = create_model_and_transforms(
        model_name, 
        pretrained, 
        force_custom_clip=True,
        skip_list=['text']
        )

    print("vision embed dim:")
    print(teacher.visual.embed_dim)
    teacher = teacher.to(f'cuda:{args.local_rank}', dtype=dtype)
    del teacher.text
    del p
    teacher = teacher.visual

    teacher.eval()  # Set the model to evaluation mode

    vit = ViT(
        image_size = max_img_size,
        patch_size = patch_size,
        dim = dim,
        heads = n_heads,
        depth = depth,
        dim_head = dim_head,
        mlp_dim = 2048,
        teacher_dim = 768,
    ).to(device, dtype=dtype)

    n_parameters = sum(p.numel() for p in vit.parameters() if p.requires_grad)

    print(f'Number of parameters: {n_parameters:,}')
    #vit.init_weights()
    postprocessors = {'bbox': PostProcess()}

    class Args:
        def __init__(self):
            self.dataset_file = 'coco'
            self.masks = False
            self.coco_path = os.getenv('COCO_PATH')
            self.eva = 1
    
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

    teacher_transform = build_teacher_transform() 

    for epoch in range(epochs):

        model_engine.train()
        imgs_processed = 0

        print("this many imgs per epoch: ")
        print(len(data_loader_train)*BS)

        for i, (samples, targets, teacher_samples) in enumerate(data_loader_train):

            samples, targets = preprocess(samples, targets, patch_size=patch_size)

            start_time = time.time()
            
            samples = samples.to(device, dtype=dtype)
            
            # Process each image in the teacher_samples tuple
            processed_teacher_samples = []
            for img in teacher_samples:
                processed_img = teacher_transform(img)
                processed_teacher_samples.append(processed_img)
            
            # Stack the processed images into a single tensor
            teacher_samples = torch.stack(processed_teacher_samples)

            target = teacher(teacher_samples.to(f'cuda:{args.local_rank}', dtype=dtype))

            out = model_engine(samples)

            loss = compute_loss(out, target)

            model_engine.backward(loss)
            model_engine.step()

            imgs_processed += BS
            
            # Perform all_reduce for each loss component
            all_reduce_start_time = time.time()
            all_reduce(loss, op=ReduceOp.SUM)
            all_reduce_end_time = time.time()
            all_reduce_time = all_reduce_end_time - all_reduce_start_time

            end_time = time.time()
            step_time = end_time - start_time
            
            loss /= world_size

            print(f'{loss:.4f} L, {step_time*1000:.4f} ms, {all_reduce_time*1000:.4f} ms all_reduce, {imgs_processed*world_size} imgs')

            if logging:
                log_dict = {
                    'epoch': epoch,
                    "loss": loss,
                    "step_time_ms": step_time * 1000,
                    "batch": i,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)

        print(f'Epoch {epoch}/{epochs} completed, {imgs_processed} images processed')

        # Save model at the end of each epoch
        save_path = f'{model_name}_epoch_{epoch}.pt'
        model_engine.save_checkpoint(save_path, epoch)
        print(f"Model saved to {save_path}")

    if logging:
        wandb.finish()

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