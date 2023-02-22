import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import glob

import torch
import torchvision.models as models
import torch.nn.functional as F

import napari

# from https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb#scrollTo=FR8YNR-g9JXA

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type(torch.FloatTensor)

# To address mps torch limitation: https://github.com/pytorch/pytorch/issues/77764
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Utils

torch_device = "cpu"

def imread(url, max_size=None, mode=None):
    if url.startswith(("http:", "https:")):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0
    return img


#

# @title VGG16 Sliced OT Style Model
vgg16 = models.vgg16(weights="IMAGENET1K_V1").features.to(torch_device)


def calc_styles_vgg(imgs):
    style_layers = [1, 6, 11, 18, 25]
    mean = torch.tensor([0.485, 0.456, 0.406], device=torch_device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=torch_device)[:, None, None]
    x = (imgs - mean) / std
    b, c, h, w = x.shape
    features = [x.reshape(b, c, h * w)]
    for i, layer in enumerate(vgg16[: max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            features.append(x.reshape(b, c, h * w))
    return features


def project_sort(x, proj):
    return torch.einsum("bcn,cp->bpn", x, proj).sort()[0]


def ot_loss(source, target, proj_n=32):
    ch, n = source.shape[-2:]
    projs = F.normalize(torch.randn(ch, proj_n, device=torch_device), dim=0)
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)
    target_interp = F.interpolate(target_proj, n, mode="nearest")
    return (source_proj - target_interp).square().sum()


def create_vgg_loss(target_img):
    yy = calc_styles_vgg(target_img)

    def loss_f(imgs):
        xx = calc_styles_vgg(imgs)
        return sum(ot_loss(x, y) for x, y in zip(xx, yy))

    return loss_f


def to_nchw(img):
    img = torch.as_tensor(img, device=torch_device)
    if len(img.shape) == 3:
        img = img[None, ...]
    return img.permute(0, 3, 1, 2)


# @title Minimalistic Neural CA
ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=torch_device)
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=torch_device)
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], device=torch_device)


def perchannel_conv(x, filters):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], "circular")
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


def perception(x):
    filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
    return perchannel_conv(x, filters)


class CA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn * 4, hidden_n, 1, device=torch_device)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False, device=torch_device)
        self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5):
        y = perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        udpate_mask = (torch.rand(b, 1, h, w, device=torch_device) + update_rate).floor()
        return x + y * udpate_mask

    def seed(self, n, sz=128):
        return torch.zeros(n, self.chn, sz, sz, device=torch_device)


def to_rgb(x):
    return x[..., :3, :, :] + 0.5


param_n = sum(p.numel() for p in CA().parameters())
print("CA param count:", param_n)

# @title Target image {vertical-output: true}
url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg"
style_img = imread(url, max_size=128)
with torch.no_grad():
    loss_f = create_vgg_loss(to_nchw(style_img))
# imshow(style_img)

# @title setup training
ca = CA()
opt = torch.optim.Adam(ca.parameters(), 1e-3, capturable=False)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 2000], 0.3)
loss_log = []
with torch.no_grad():
    pool = ca.seed(256)

## Training

viewer = napari.Viewer()

# @title training loop {vertical-output: true}

gradient_checkpoints = False  # Set in case of OOM problems

for i in range(5000):
    with torch.no_grad():
        batch_idx = np.random.choice(len(pool), 4, replace=False)
        x = pool[batch_idx]
        if i % 8 == 0:
            x[:1] = ca.seed(1)
    step_n = np.random.randint(32, 96)
    if not gradient_checkpoints:
        for k in range(step_n):
            x = ca(x)
    else:
        x.requires_grad = True  # https://github.com/pytorch/pytorch/issues/42812
        x = torch.utils.checkpoint.checkpoint_sequential([ca] * step_n, 16, x)

    overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
    loss = loss_f(to_rgb(x)) + overflow_loss
    with torch.no_grad():
        loss.backward()
        for p in ca.parameters():
            p.grad /= p.grad.norm() + 1e-8  # normalize gradients
        opt.step()
        opt.zero_grad()
        lr_sched.step()
        pool[batch_idx] = x  # update pool

        loss_log.append(loss.item())
        if i % 5 == 0:
            print(
                f"""
        step_n: {len(loss_log)}
        loss: {loss.item()}
        lr: {lr_sched.get_last_lr()[0]}"""
            )
            imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
            viewer.add_image(np.hstack(imgs))
        if i % 50 == 0:
            pl.plot(loss_log, ".", alpha=0.1)
            pl.yscale("log")
            pl.ylim(np.min(loss_log), loss_log[0])
            pl.tight_layout()
            # imshow(grab_plot(), id='log')            
            # imshow(np.hstack(imgs), id='batch')
