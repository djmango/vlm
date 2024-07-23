import sys
import os
detr_path = os.path.join(os.path.dirname(__file__), 'detr')
vit_path = os.path.join(os.path.dirname(__file__), 'vit-pytorch')
sys.path.append(detr_path)
sys.path.append(vit_path)
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from vit_pytorch.na_vit import NaViT
from detr.models.detr import SetCriterion
from detr.models.matcher import build_matcher
from data import apply_bbox_to_image
from PIL import Image
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def normalize_coords(x, img_sizes):
    x0, y0, x1, y1 = x.unbind(-1)
    imgw, imgh = img_sizes.unbind(-1)
    if x.dim() == 3: imgw, imgh = imgw[:, None], imgh[:, None]
    return torch.stack([x0 / imgw, y0 / imgh, x1 / imgw, y1 / imgh], dim=-1)

def unnormalize_coords(x, img_sizes):
    cx, cy, w, h = x.unbind(-1)
    imgw, imgh = img_sizes.unbind(-1)
    if x.dim() == 3: imgw, imgh = imgw[:, None], imgh[:, None]
    return torch.stack([cx * imgw, cy * imgh, w * imgw, h * imgh], dim=-1)

def display_img(img, bboxes, desc='sample_bbox'):
    img_pil = transforms.ToPILImage()(img.cpu())
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.view(-1, 4).tolist()
    for bbox in bboxes:
        img_pil = apply_bbox_to_image(img_pil, bbox)
    os.makedirs('vis', exist_ok=True)
    img_pil.save(f'vis/{desc}.png')

# Configuration
patch_size = 32
max_img_size = patch_size * 200
n_classes = 88
n_bboxs = 5
dim = 1024
n_heads = 2
depth = 16

# Loss config
CLS_WEIGHT = 0.0
GIOU_WEIGHT = 1.0
L1_WEIGHT = 5.0
EOS_CONF = 0.0

class ImageProcessor:
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.resolution_map = {
            r'default_(\d+)-(\d+)': lambda w, h: (int(w), int(h)),
            'iPad-Pro': (1138, 1518),
            'iPhone-13 Pro': (650, 1407)
        }
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),
        ])

    def get_target_resolution(self, key):
        for pattern, resolution in self.resolution_map.items():
            if isinstance(resolution, tuple):
                if key == pattern:
                    return resolution
            else:
                match = re.match(pattern, key)
                if match:
                    return resolution(*match.groups())
        raise ValueError(f"Unknown key: {key}")

    def get_scale_factors(self, original_size, key):
        target_w, target_h = self.get_target_resolution(key)
        wildcard = 3 if key == 'iPhone-13 Pro' else (2 if key == 'iPad-Pro' else 1)
        return target_w / original_size[0] * wildcard, target_h / original_size[1] * wildcard

    def rescale_image(self, img, key):
        c, h, w = img.shape
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

    def process_image(self, item):
        image = self.rescale_image(self.transform(item['image']), item['key_name']).to(device)
        content_boxes = [bbox for bbox in item['contentBoxes'] if bbox[2] >= bbox[0] and bbox[3] >= bbox[1]]
        scale_w, scale_h = self.get_scale_factors(item['image'].size, item['key_name'])
        scaled_boxes = [self.scale_bbox(box, scale_w, scale_h) for box in content_boxes]
        return image, scaled_boxes, item['labels']

def main():
    # Load dataset and get first 3 images
    ds = load_dataset("biglab/webui-7k-elements")
    first_3_samples = ds['train'][:3]

    print(first_3_samples)

    # Process images
    processor = ImageProcessor(patch_size)
    processed_samples = []
    for sample in first_3_samples:
        image = processor.rescale_image(processor.transform(sample['image']), sample['key_name']).to(device)
        content_boxes = [bbox for bbox in sample['contentBoxes'] if bbox[2] >= bbox[0] and bbox[3] >= bbox[1]]
        scale_w, scale_h = processor.get_scale_factors(sample['image'].size, sample['key_name'])
        scaled_boxes = [processor.scale_bbox(box, scale_w, scale_h) for box in content_boxes]
        processed_samples.append((image, scaled_boxes, sample['labels']))

    # Initialize model
    vit = NaViT(
        image_size=max_img_size,
        patch_size=patch_size,
        n_bboxs=n_bboxs,
        n_classes=n_classes,
        dim=dim,
        heads=n_heads,
        depth=depth,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        token_dropout_prob=0.1
    ).to(device)

    # Prepare criterion
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': CLS_WEIGHT, 'loss_bbox': L1_WEIGHT, 'loss_giou': GIOU_WEIGHT}
    matcher = build_matcher(cost_class=CLS_WEIGHT, cost_bbox=L1_WEIGHT, cost_giou=GIOU_WEIGHT)
    criterion = SetCriterion(n_classes, matcher, weight_dict, EOS_CONF, losses).to(device)

    optimizer = torch.optim.Adam(vit.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        vit.train()
        total_loss = 0

        for i, (img, bboxes, labels) in enumerate(processed_samples):
            img = img.unsqueeze(0)  # Add batch dimension

            # Prepare target
            target_dict = {}
            sampled_bboxes = torch.tensor(bboxes[:n_bboxs], device=device)
            sampled_labels = labels[:n_bboxs]
            
            bbox_target = torch.cat([sampled_bboxes, torch.zeros(n_bboxs - len(sampled_bboxes), 4, device=device)])
            bbox_target = box_xyxy_to_cxcywh(bbox_target)
            img_wh = torch.tensor([img.shape[3], img.shape[2]])
            normalized_bbox_target = normalize_coords(bbox_target, img_wh)

            cls_to_idx = {cls: idx for idx, cls in enumerate(set(label[0] for label in sampled_labels))}
            label_indices = torch.tensor([cls_to_idx[label[0]] for label in sampled_labels], dtype=torch.long, device=device)
            padded_labels = torch.full((n_bboxs,), n_classes, dtype=torch.long, device=device)
            padded_labels[:len(label_indices)] = label_indices

            target_dict['boxes'] = normalized_bbox_target.unsqueeze(0)  # Add batch dimension
            target_dict['labels'] = padded_labels.unsqueeze(0)  # Add batch dimension

            # Forward pass
            out_cls, out_bbox = vit([img])

            out_cls = out_cls.view(1, n_bboxs, n_classes + 1)
            out_bbox = out_bbox.view(1, n_bboxs, 4)

            outs = {'pred_logits': out_cls, 'pred_boxes': out_bbox}

            # Compute loss
            loss_dict = criterion(outs, [target_dict])
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            # Visualize results every 100 epochs
            if epoch % 100 == 0:
                pred_bboxes = box_cxcywh_to_xyxy(unnormalize_coords(out_bbox.squeeze(0), img_wh))
                display_img(img.squeeze(0), pred_bboxes, desc=f"pred_{i}_epoch_{epoch}")
                display_img(img.squeeze(0), sampled_bboxes, desc=f"target_{i}_epoch_{epoch}")

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    print("Training completed.")

if __name__ == '__main__':
    main()