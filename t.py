import torch
import sys, os

eva_path = os.path.join(os.path.dirname(__file__), 'EVA/EVA-CLIP/rei')
sys.path.append(eva_path)

from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

model_name = "EVA02-CLIP-B-16" 
pretrained = "/workspace/vlm/EVA02_CLIP_B_psz16_s8B.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

image_path = "/workspace/vlm/cat.jpeg"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = create_model_and_transforms(
    model_name, 
    pretrained, 
    force_custom_clip=True,
    pretrained_text='',
    skip_list=['text']
    )

tokenizer = get_tokenizer(model_name)
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

print(iamge_features.shape)

