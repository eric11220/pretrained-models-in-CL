import torch
from pl_bolts.models.self_supervised import SimCLR, SwAV
from torch import nn

from .pretrained import SelfSupResnetWrapper

def get_swav_rn50():
    dim_in, nclass = 2048, 100
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    model = SwAV.load_from_checkpoint(weight_path, strict=True)
    model = nn.Sequential(*list(list(model.children())[0].children())[:-2])
    return SelfSupResnetWrapper(nclass, model, dim_in=dim_in)

def get_simclr_rn50():
    dim_in, nclass = 2048, 100
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    model = simclr.encoder
    model = nn.Sequential(*list(model.children())[:-1])
    return SelfSupResnetWrapper(nclass, model, dim_in=dim_in)

def get_barlow_twins_rn50():
    dim_in, nclass = 2048, 100
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return SelfSupResnetWrapper(nclass, model, dim_in=dim_in)
