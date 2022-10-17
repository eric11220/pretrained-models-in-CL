from abc import abstractmethod
import abc
import numpy as np
import torch

from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, setup_opt

from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle


class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.fix_bn = params.fix_bn
        self.arch = params.model
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

        if not self.fix_bn: self.model = self.model.train()
        else: self.model = self.model.eval()

        return self.get_train_loader(x_train, y_train)

    def get_train_loader(self, x_train, y_train):
        train_dataset = dataset_transform(x_train, y_train,
                                          transform=transforms_match(model=self.arch, reshape224=self.params.reshape224)[self.data])
        train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=4,
                                       drop_last=True)
        return train_loader

    def train_learner(self, x_train, y_train, offline_epoch=0):
        train_loader = self.before_train(x_train, y_train)
        self.train(train_loader, epoch=self.epoch,
                   mem=self.buffer if hasattr(self, 'buffer') else None,
                   mem_iters=self.mem_iters if hasattr(self, 'mem_iters') else None)

        if offline_epoch > 0 and hasattr(self, 'buffer'):
            print(f'Training offline on memory for {offline_epoch} epochs')
            mem_set = TensorDataset(self.buffer.buffer_img, self.buffer.buffer_label)
            train_loader = DataLoader(mem_set, batch_size=self.batch, shuffle=True, num_workers=4, drop_last=True)

            opt_holder = self.opt
            self.opt = setup_opt(self.params.optimizer, self.model, self.params.offline_lr, self.params.weight_decay)
            self.train(train_loader, epoch=offline_epoch, mem_iters=1, mem=None)
            self.opt = opt_holder

        self.after_train()

    @abstractmethod
    def train(self, train_loader, epoch, mem_iters, mem=None, print_freq=100):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            if not self.fix_bn: self.model.train()
            else: self.model.eval()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def get_feature(self, x):
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        feature = model.features(x).detach().clone()
        return feature

    def get_logits(self, x):
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        feature = model.logits(x).detach().clone()
        return feature

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)

    def evaluate(self, test_loaders, compute_unif=False):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                x, y = maybe_cuda(x), maybe_cuda(y)
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    # feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = self.get_feature(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.get_feature(ex.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y

        all_features = []
        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                        # feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        feature = self.get_feature(batch_x) # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                        #old ncm
                        means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        means = means.transpose(1, 2)
                        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        _, pred_label = dists.min(1)
                        # may be faster
                        # feature = feature.squeeze(2).T
                        # _, preds = torch.matmul(means, feature).max(0)
                        correct_cnt = (np.array(self.old_labels)[
                                           pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    else:
                        feature = self.get_feature(batch_x) # (batch_size, feature_size)
                        logits = self.get_logits(feature)
                        _, pred_label = torch.max(logits, 1)
                        correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                    if compute_unif:
                        all_features.append(feature)

                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen-1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                        elif task == self.task_seen -1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                acc_array[task] = acc.avg()

        if compute_unif:
            all_features = torch.cat(all_features, dim=0)
            all_features = F.normalize(all_features, dim=-1)
            print(all_features.norm(dim=-1))
            uniformity = torch.pdist(all_features, p=2).pow(2).mul(-2).exp().mean().log()
            print(f'Uniformity: {uniformity}')

        print(acc_array)
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
            with open('confusion', 'wb') as fp:
                pickle.dump([correct_lb, predict_lb], fp)
        return acc_array
