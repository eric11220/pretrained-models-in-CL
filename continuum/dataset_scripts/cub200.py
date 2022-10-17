import numpy as np
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

from .cub2011 import Cub2011


class CUB200(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'cub200'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(CUB200, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)
        self.class_order = params.class_order


    def download_load(self):
        dataset_train = Cub2011(root=self.root, train=True, download=True)
        self.train_data = dataset_train.data
        self.train_label = np.array(dataset_train.targets)
        dataset_test = Cub2011(root=self.root, train=False, download=True)
        self.test_data = dataset_test.data
        self.test_label = np.array(dataset_test.targets)

    def setup(self):
        if self.scenario == 'ni':
            raise Exception('Not implemented')
        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(
                    class_nums=200, num_tasks=self.task_nums, fixed_order=self.params.fix_order, class_order=self.class_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            raise Exception('Not implemented')
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                                                         self.params.ns_factor)
