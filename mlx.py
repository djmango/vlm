import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
import random
import json
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from train import NaViTDataset

# Helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# Layer implementations

class LayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.weight = mx.ones(dims)
        self.bias = mx.zeros(dims)

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / mx.sqrt(var + 1e-5) + self.bias

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, 1, dim))

    def __call__(self, x):
        normed = x / mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + 1e-8)
        return normed * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def __call__(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def __call__(self, x, context=None, mask=None):
        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = mx.split(self.to_kv(kv_input), 2, axis=-1)

        q = mx.reshape(q, (q.shape[0], -1, self.heads, q.shape[-1] // self.heads))
        k = mx.reshape(k, (k.shape[0], -1, self.heads, k.shape[-1] // self.heads))
        v = mx.reshape(v, (v.shape[0], -1, self.heads, v.shape[-1] // self.heads))

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = mx.matmul(q, k.transpose(0, 2, 1, 3))

        if exists(mask):
            mask = mx.expand_dims(mask, axis=(1, 2))
            dots = mx.where(mask, dots, mx.finfo(dots.dtype).min)

        attn = nn.softmax(dots, axis=-1)
        out = mx.matmul(attn, v)
        out = mx.reshape(out, (out.shape[0], -1, out.shape[-2] * out.shape[-1]))
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = [
            (Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
             FeedForward(dim, mlp_dim, dropout=dropout))
            for _ in range(depth)
        ]
        self.norm = LayerNorm(dim)

    def __call__(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.norm(x)

class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, n_bboxs, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0., token_dropout_prob=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_size
        patch_dim = channels * (patch_size ** 2)

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = mx.random.normal((image_height // patch_size, dim))
        self.pos_embed_width = mx.random.normal((image_width // patch_size, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.attn_pool_queries = mx.random.normal((dim,))
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, n_bboxs * 5, bias=False)
        )

    def __call__(self, x):
        b, c, h, w = x.shape
        p = self.patch_size

        x = mx.reshape(x, (b, c, h // p, p, w // p, p))
        x = mx.transpose(x, (0, 2, 4, 3, 5, 1))
        x = mx.reshape(x, (b, (h // p) * (w // p), c * p * p))

        x = self.to_patch_embedding(x)

        h_indices = mx.arange(h // p).reshape(1, -1, 1).repeat(b, axis=0).repeat(w // p, axis=1)
        w_indices = mx.arange(w // p).reshape(1, 1, -1).repeat(b, axis=0).repeat(h // p, axis=1)

        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos
        x = self.dropout(x)
        x = self.transformer(x)

        queries = mx.broadcast_to(self.attn_pool_queries, (b, 1, x.shape[-1]))
        x = self.attn_pool(queries, context=x)
        x = mx.reshape(x, (b, -1))

        x = self.to_latent(x)
        return self.mlp_head(x)


def rescale_image(img, target_size):
    w, h = img.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    new_img.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.array(new_img)

def get_batch(dataset, batch_size, patch_size, n_bboxs):
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    batch = []
    for idx in indices:
        item = dataset[idx]
        img = Image.open(item['image_path']).convert('RGB')
        img = rescale_image(img, (224, 224))  # Resize to a fixed size for simplicity
        img = np.transpose(img, (2, 0, 1)) / 255.0  # Normalize to [0, 1]
        
        bboxes = item['contentBoxes']
        sampled_bboxes = random.sample(bboxes, min(n_bboxs, len(bboxes)))
        
        confidence_target = [1] * len(sampled_bboxes) + [0] * (n_bboxs - len(sampled_bboxes))
        bbox_target = sampled_bboxes + [[0,0,0,0]] * (n_bboxs - len(sampled_bboxes))
        
        batch.append((img, np.array(confidence_target), np.array(bbox_target)))
    
    return map(mx.array, zip(*batch))

# Loss function

def loss_fn(confidence_pred, bbox_pred, confidence_target, bbox_target):
    confidence_loss = nn.binary_cross_entropy_with_logits(confidence_pred, confidence_target)
    positive_mask = confidence_target > 0
    bbox_loss = mx.mean((bbox_pred[positive_mask] - bbox_target[positive_mask]) ** 2)
    return confidence_loss + bbox_loss

# Training loop

def train(model, dataset, num_epochs, batch_size, learning_rate):
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    @mx.compile
    def step(model, images, confidence_targets, bbox_targets):
        def loss(model):
            preds = model(images)
            preds = mx.reshape(preds, (preds.shape[0], -1, 5))
            confidence_preds = preds[:, :, 0]
            bbox_preds = preds[:, :, 1:]
            return loss_fn(confidence_preds, bbox_preds, confidence_targets, bbox_targets)
        
        grad_fn = mx.grad(loss)
        grads = grad_fn(model)
        optimizer.update(model, grads)
        return loss(model)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = len(dataset) // batch_size
        for _ in range(num_batches):
            images, confidence_targets, bbox_targets = get_batch(dataset, batch_size, model.patch_size, 100)
            loss = step(model, images, confidence_targets, bbox_targets)
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}")

# Main execution

def main():
    # Hyperparameters
    image_size = 224
    patch_size = 16
    n_bboxs = 100
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 1024
    channels = 3
    num_epochs = 10
    batch_size = 32
    learning_rate = 3e-4

    # Initialize model
    model = NaViT(
        image_size=image_size,
        patch_size=patch_size,
        n_bboxs=n_bboxs,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels
    )

    # Load dataset
    ds = load_dataset("biglab/webui-7kbal-elements")
    dataset = NaViTDataset(ds['train'], patch_size, 14400)

    # Train the model
    train(model, dataset, num_epochs, batch_size, learning_rate)

if __name__ == "__main__":
    main()