import torch
import sys, os

eva_path = os.path.join(os.path.dirname(__file__), 'EVA/EVA-CLIP/rei')
vit_path = os.path.join(os.path.dirname(__file__), 'vit-pytorch')
sys.path.append(eva_path)

from eva_clip import create_model_and_transforms, get_tokenizer
from vit_pytorch.det_vit import ViT
from PIL import Image


def test_clip():
    #model_name = "EVA02-CLIP-B-16" 
    #pretrained = "/workspace/vlm/EVA02_CLIP_B_psz16_s8B.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
    #pretrained = '/path/to/EVA02_CLIP_B_psz16_s8B.pt'

    model_name = "EVA02-CLIP-bigE-14-plus" 
    pretrained = "/workspace/vlm/EVA02_CLIP_E_psz14_plus_s9B.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"


    image_path = "/workspace/vlm/cat.jpeg"
    caption = ["a diagram", "a dog", "a cat"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, p = create_model_and_transforms(
        model_name, 
        pretrained, 
        force_custom_clip=True,
        pretrained_text='',
        skip_list=['text']
        )
    BS = 1
    img_size = 224
    patch_size = 14

    seq_len = (img_size//patch_size) ** 2
    mask_ratio = 0.4
    bool_mask = torch.rand(BS, seq_len) < mask_ratio
    
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=torch.float16)
    del model.text
    print('loading done')
    input()
    image = p(Image.open(image_path)).unsqueeze(0).to(device, dtype=torch.float16)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image, bool_masked_pos=bool_mask)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    print(image_features.shape)

def test_load_eva():
    path = '/workspace/vlm/eva_coco_checkpoint_epoch_4.pt/eva_coco_epoch_4.pt'

    patch_size = 16
    max_img_size = 1440
    num_bboxs = 100
    num_classes = 91
    dim_head = 64
    n_heads = 2
    dim = 1024
    class_head_dim = int(dim * 2)
    depth = 10
    epochs = 300  # As per DETR paper
    dtype = torch.float16

    # loss config
    EOS_CONF = 0.1
    CLS_WEIGHT = 1.0
    GIOU_WEIGHT = 2.0
    L1_WEIGHT = 5.0

    vit = ViT(
        image_size = max_img_size,
        patch_size = patch_size,
        num_bboxs = num_bboxs,
        num_classes = num_classes,
        dim = dim,
        heads = n_heads,
        depth = depth,
        dim_head = dim_head,
        mlp_dim = 2048,
        class_head_dim = int(dim * 2)
    ).to('cuda', dtype=torch.float16)

    return load_eva_ckpt(path, vit)

def load_eva_ckpt(path, vit, keys_to_del=[]):
    checkpoint = torch.load(path, map_location='cuda')
    
    for key in list(checkpoint.keys()):
        if any(k in key for k in keys_to_del):
            del checkpoint[key]

    vit.load_state_dict(checkpoint, strict=False)


a = torch.tensor([[ 0.9041,  0.0196],
        [-0.3108, -2.4423],
        [-0.4821,  1.0590]])

b = torch.tensor([[-2.1763, -0.4713],
        [-0.6986,  1.3702]])

c = torch.cdist(a, b, p=1)

def calc_p1(a, b):
    return torch.sum(torch.abs(a[:, None, :] - b[None, :, :]), dim=2)
    
#print(c)
#print(calc_p1(a,b))
test_clip()




    
#test_load_eva()