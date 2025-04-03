# import os
# import torch
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
# import timm
# from huggingface_hub import login, hf_hub_download

# login("")  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# local_dir = ""
# os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
# timm_kwargs = {
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#         }
# model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
# model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
# transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
# model.eval()

# import os
# import torch
# from torchvision import transforms
# import timm
# from huggingface_hub import login, hf_hub_download

# login("")  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# local_dir = "/blue/cis6930/reardons/CANCER-TRANS-2025"
# os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
# model = timm.create_model(
#     "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
# )
# model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
# transform = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]
# )
# model.eval()

import os
import torch
import timm
from huggingface_hub import login, hf_hub_download
from timm.data import resolve_data_config, create_transform

# Authentication
login(token="")

# Model download
local_dir = "/blue/cis6930/reardons/CANCER-TRANS-2025"
os.makedirs(local_dir, exist_ok=True)
hf_hub_download(
    "MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir=local_dir,
    force_download=True
)

# Model initialization
model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
    dynamic_img_pad=True,
    num_classes=0
)

# Load weights
model.load_state_dict(
    torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
    strict=True
)

# TIMM transforms
data_config = resolve_data_config(model.pretrained_cfg, model=model)
transform = create_transform(**data_config)

model.eval()
