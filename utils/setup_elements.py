import numpy as np
import torch
import PIL
from models.resnet import Reduced_ResNet18, SupConResNet
from models.pretrained import ResNet_standard
from models.self_sup_model import get_swav_rn50, get_simclr_rn50, get_barlow_twins_rn50
from models.clip_encoder import ClipImageEncoder
from torchvision import transforms
import torch.nn as nn

from pl_bolts.models.self_supervised import SimCLR, SwAV

from .class_names import cifar100_classes, mini_imagenet_classes, cub200_classes


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


def input_size_match(model=None, reshape224=False):
    size_match = {
        'cifar100': [3, 32, 32],
        'cifar10': [3, 32, 32],
        'cub200': [3, 224, 224],
        'core50': [3, 128, 128],
        'mini_imagenet': [3, 84, 84],
        'openloris': [3, 50, 50]
    }
    if 'pretrained' in model or model in ['clip', 'swav', 'simclr', 'barlow_twins'] or reshape224:
        size_match = {
            'cifar100': [3, 224, 224],
            'cifar10': [3, 224, 224],
            'cub200': [3, 224, 224],
            'core50': [3, 224, 224],
            'mini_imagenet': [3, 224, 224],
            'openloris': [3, 224, 224]
        }
    # print(size_match['cifar100'])
    return size_match


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'cub200': 200,
    'core50': 50,
    'mini_imagenet': 100,
    'openloris': 69
}

classes = {
    'cifar100': cifar100_classes,
    'cub200': cub200_classes,
    'mini_imagenet': mini_imagenet_classes
}

def transforms_match(model=None, reshape224=False, test=False):
    im_size = None
    mean, std = (0., 0., 0.), (1., 1., 1.)

    if 'pretrained' in model or model in ['swav', 'simclr', 'barlow_twins'] or reshape224:
        # ImageNet stats
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        im_size = 224
    elif model == 'clip':
        # CLIP stats
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        im_size = 224

    if reshape224: im_size = 224

    match = {
        'core50': transforms.Compose([
            transforms.ToTensor(),
            ]),
        'cifar100': transforms.Compose([
            transforms.Resize(im_size if im_size is not None else 32, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        'cub200': transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(224) if not test else transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        'cifar10': transforms.Compose([
            transforms.ToTensor(),
            ]),
        'mini_imagenet': transforms.Compose([
            transforms.Resize(im_size if im_size is not None else 84, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        'openloris': transforms.Compose([
                transforms.ToTensor()])
    }

    # print(match['cifar100'])
    return match


def setup_architecture(params, class_order=None):
    nclass = n_classes[params.data]
    class_name = classes[params.data]
    if params.agent == 'ICARL':
        class_order = np.array(class_order)
        class_order = np.array([arr[ind] for arr, ind in zip(class_order, np.argsort(class_order, axis=-1))])
        class_order = class_order.reshape(-1)
        class_name = np.array(class_name)[class_order]

    if params.agent in ['SCR', 'SCP']:
        dim_in = 640 if params.data == 'mini_imagenet' else 160
        return SupConResNet(dim_in=dim_in, head=params.head, model=params.model, class_name=class_name)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)

    feat_dim = None
    if params.data == 'mini_imagenet':
        feat_dim = 640

    if params.model == 'reduced_rn18':
        return Reduced_ResNet18(nclass, feat_dim=feat_dim, norm=params.feat_norm)
    elif 'pretrained' in params.model:
        rn = 18 if 'rn18' in params.model else 50
        return ResNet_standard(nclass, pretrained=True, rn=rn)
    elif params.model == 'rn18':
        return ResNet_standard(nclass, pretrained=False, rn=18)
    elif params.model == 'clip':
        return ClipImageEncoder(nclass, class_name)
    elif params.model == 'simclr':
        return get_simclr_rn50()
    elif params.model == 'swav':
        return get_swav_rn50()
    elif params.model == 'barlow_twins':
        return get_barlow_twins_rn50()
    else:
        raise Exception('Unsupported architecture')


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
