from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from torch.utils import data
from utils.buffer.buffer import Buffer
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import torch


class AGEM(ContinualLearner):
    def __init__(self, model, opt, params):
        super(AGEM, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

    def train(self, train_loader, epoch, mem_iters=1, mem=None):
        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(mem_iters):
                    logits = self.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                    self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    if self.task_seen > 0 and mem is not None:
                        # sample from memory of previous tasks
                        mem_x, mem_y = self.buffer.retrieve()
                        if mem_x.size(0) > 0:
                            params = [p for p in self.model.parameters() if p.requires_grad]
                            # gradient computed using current batch
                            grad = [p.grad.clone() for p in params]
                            mem_x = maybe_cuda(mem_x, self.cuda)
                            mem_y = maybe_cuda(mem_y, self.cuda)
                            mem_logits = self.forward(mem_x)
                            loss_mem = self.criterion(mem_logits, mem_y)
                            self.opt.zero_grad()
                            loss_mem.backward()
                            # gradient computed using memory samples
                            grad_ref = [p.grad.clone() for p in params]

                            # inner product of grad and grad_ref
                            prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
                            if prod < 0:
                                prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
                                # do projection
                                grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
                            # replace params' grad
                            for g, p in zip(grad, params):
                                p.grad.data.copy_(g)
                    self.opt.step()

                if mem is not None:
                    # update mem
                    self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
