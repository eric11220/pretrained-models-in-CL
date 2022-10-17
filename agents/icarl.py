from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils import utils
from utils.buffer.buffer_utils import random_retrieve
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.nn import functional as F
from utils.utils import maybe_cuda, AverageMeter
from utils.buffer.buffer import Buffer
import torch
import copy


class Icarl(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Icarl, self).__init__(model, opt, params)
        self.model = model
        self.mem_size = params.mem_size
        self.buffer = Buffer(model, params)
        self.prev_model = None

    def train(self, train_loader, epoch, mem_iters, mem=None, print_freq=100):
        if mem is None:
            prev_model_holder = self.prev_model

        self.update_representation(train_loader, epoch, mem)
        self.prev_model = copy.deepcopy(self.model)

    def update_representation(self, train_loader, epoch, mem):
        updated_idx = []
        for ep in range(epoch):
            for i, train_data in enumerate(train_loader):
                # batch update
                train_x, train_y = train_data
                train_x = maybe_cuda(train_x, self.cuda)
                train_y = maybe_cuda(train_y, self.cuda)
                train_y_copy = train_y.clone()
                for k, y in enumerate(train_y_copy):
                    if y in self.new_labels: # Belongs to new class
                        train_y_copy[k] = len(self.old_labels) + self.new_labels.index(y)
                    else: # old class
                        train_y_copy[k] = self.old_labels.index(y)

                all_cls_num = len(self.new_labels) + len(self.old_labels)
                target_labels = utils.ohe_label(train_y_copy, all_cls_num, device=train_y_copy.device).float()
                if self.prev_model is not None and mem is not None:
                    mem_x, mem_y = random_retrieve(mem, self.batch,
                                                   excl_indices=updated_idx)
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    batch_x = torch.cat([train_x, mem_x])
                    target_labels = torch.cat([target_labels, torch.zeros_like(target_labels)])
                else:
                    batch_x = train_x
                logits = self.forward(batch_x)
                self.opt.zero_grad()
                if self.prev_model is not None:
                    with torch.no_grad():
                        q = torch.sigmoid(self.prev_model.forward(batch_x))
                    for k, y in enumerate(self.old_labels):
                        target_labels[:, k] = q[:, k]
                loss = F.binary_cross_entropy_with_logits(logits[:, :all_cls_num], target_labels, reduction='none').sum(dim=1).mean()
                loss.backward()
                self.opt.step()
                if mem is not None:
                    updated_idx += mem.update(train_x, train_y)
