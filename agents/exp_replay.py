import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

    def train(self, train_loader, epoch, mem_iters, mem=None, print_freq=100):
        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(mem_iters):
                    logits = self.model.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    if mem is not None:
                        mem_x, mem_y = mem.retrieve(x=batch_x, y=batch_y)
                        if mem_x.size(0) > 0:
                            mem_x = maybe_cuda(mem_x, self.cuda)
                            mem_y = maybe_cuda(mem_y, self.cuda)
                            mem_logits = self.model.forward(mem_x)
                            loss_mem = self.criterion(mem_logits, mem_y)
                            if self.params.trick['kd_trick']:
                                loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                           self.kd_manager.get_kd_loss(mem_logits, mem_x)
                            if self.params.trick['kd_trick_star']:
                                loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                       (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                             mem_x)
                            # update tracker
                            losses_mem.update(loss_mem, mem_y.size(0))
                            _, pred_label = torch.max(mem_logits, 1)
                            correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                            acc_mem.update(correct_cnt, mem_y.size(0))

                            loss_mem.backward()

                        if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                            # opt update
                            self.opt.zero_grad()
                            combined_batch = torch.cat((mem_x, batch_x))
                            combined_labels = torch.cat((mem_y, batch_y))
                            combined_logits = self.model.forward(combined_batch)
                            loss_combined = self.criterion(combined_logits, combined_labels)
                            loss_combined.backward()
                            self.opt.step()
                        else:
                            self.opt.step()
                    else:
                        self.opt.step()

                if mem is not None:
                    # update mem
                    self.buffer.update(batch_x, batch_y)

                if i % print_freq == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg()),
                        end='\r'
                    )
                    if mem is not None:
                        print(
                            '==>>> it: {}, mem avg. loss: {:.6f}, '
                            'running mem acc: {:.3f}'
                                .format(i, losses_mem.avg(), acc_mem.avg())
                        )
