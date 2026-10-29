"""
resnet12.py — drop into the DINO repo root alongside main_dino.py.

ResNet-12 backbone for few-shot learning, compatible with DINO's MultiCropWrapper.
MultiCropWrapper sets backbone.fc = backbone.head = nn.Identity() at init;
ResNet12.forward() does not call either, so this is harmless.

Changes required in main_dino.py:
  1. Add at the top (with the other imports):
         from src.core import resnet12 as resnet12_models

  2. In train_dino(), add this elif BEFORE the torchvision fallback:
         elif args.arch in resnet12_models.__dict__:
             student = resnet12_models.__dict__[args.arch]()
             teacher = resnet12_models.__dict__[args.arch]()
             embed_dim = student.embed_dim

  3. Run with:
         python -m torch.distributed.launch --nproc_per_node=1 main_dino.py \
           --arch resnet12 \
           --data_path /path/to/raw/androFIM_pretrain \
           --output_dir /path/to/models/androfim \
           --epochs 100 \
           --batch_size_per_gpu 64 \
           --local_crops_number 6 \
           --global_crops_scale 0.4 1.0 \
           --local_crops_scale 0.05 0.4
"""

import torch.nn as nn


class ResNet12Block(nn.Module):
    """Residual block: 3×Conv→BN→LeakyReLU with 1×1 shortcut, MaxPool, Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act     = nn.LeakyReLU(0.1, inplace=True)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.act(x + res)
        return self.dropout(self.pool(x))


class ResNet12(nn.Module):
    """
    ResNet-12 (channels: 64→128→256→512).
    Returns flat (B, 512) features; no classification head.
    embed_dim attribute is read by main_dino.py to size the DINOHead.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.layer1  = ResNet12Block(3,   64,  dropout)
        self.layer2  = ResNet12Block(64,  128, dropout)
        self.layer3  = ResNet12Block(128, 256, dropout)
        self.layer4  = ResNet12Block(256, 512, dropout)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = 512

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)


def resnet12(**kwargs):
    return ResNet12(**kwargs)
