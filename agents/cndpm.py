from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from models.ndpm.ndpm import Ndpm
from torch import nn
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch


class Cndpm(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Cndpm, self).__init__(model, opt, params)
        self.model = model


    def train(self, train_loader, epoch, mem_iters, mem=None, print_freq=100):
        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.learn(batch_x, batch_y)
                else:
                    self.model.learn(batch_x, batch_y)
                if self.params.verbose:
                    if isinstance(self.model, nn.DataParallel):
                        print('\r[Step {:4}] STM: {:5}/{} | #Expert: {}'.format(
                            i,
                            len(self.model.module.stm_x), self.params.stm_capacity,
                            len(self.model.module.experts) - 1
                        ), end='')
                    else:
                        print('\r[Step {:4}] STM: {:5}/{} | #Expert: {}'.format(
                            i,
                            len(self.model.stm_x), self.params.stm_capacity,
                            len(self.model.experts) - 1
                        ), end='')
        print()
