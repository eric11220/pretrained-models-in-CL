import time
import torch
import numpy as np
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
from torch import nn
import pandas as pd
import os
import pickle


def multiple_run(params, store=False, save_path=None, store_mem_snapshot=False):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = os.path.join(table_path, params.exp_name)
            os.makedirs(save_path, exist_ok=True)
            save_stat_path = os.path.join(save_path, 'stats.pkl')

        # Experiments have been run
        if os.path.isfile(save_stat_path) or os.path.isfile(os.path.join(table_path, f'{params.exp_name}.pkl')):
            print(f'{params.exp_name} has run before, exiting')
            return

    results = []
    for run in range(params.num_runs):
        accuracy_list = []
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()

        if params.target_run is not None:
            save_stat_path = os.path.join(save_path, f'stats_run{params.target_run}.pkl')
            if run != params.target_run:
                continue
            else:
                print(f'Will be saving results to {save_stat_path}')

        model = setup_architecture(params, class_order=data_continuum.data_object.task_labels)
        model = maybe_cuda(model, params.cuda)
        model = nn.DataParallel(model)
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        agent = agents[params.agent](model, opt, params)

        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        if params.online:
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                print("-----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train,
                                    offline_epoch=params.offline_epoch if i == data_continuum.task_nums-1 else 0)
                acc_array = agent.evaluate(test_loaders, compute_unif=(i==data_continuum.task_nums-1 and params.cal_uniformity))
                tmp_acc.append(acc_array)

                if params.store_checkpoint:
                    agent.save_model(os.path.join(save_path, f'task{i}.pt'))
                if params.store_mem_snapshot and hasattr(agent, 'buffer'):
                    mem_snapshot = {
                        'mem_x': agent.buffer.buffer_img[:agent.buffer.current_index],
                        'mem_y': agent.buffer.buffer_label[:agent.buffer.current_index]
                    }
                    snapshot_path = os.path.join(save_path, f'mem_task{i}.pt')
                    torch.save(mem_snapshot, snapshot_path)

            run_end = time.time()
            print(
                "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                               run_end - run_start))
            accuracy_list.append(np.array(tmp_acc))
        else:
            x_train_offline = []
            y_train_offline = []
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
            print('Training Start')
            x_train_offline = np.concatenate(x_train_offline, axis=0)
            y_train_offline = np.concatenate(y_train_offline, axis=0)
            print("----------run {} training-------------".format(run))
            print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
            agent.train_learner(x_train_offline, y_train_offline)
            acc_array = agent.evaluate(test_loaders, compute_unif=(i==data_continuum.task_nums-1 and args.cal_uniformity))
            accuracy_list.append(acc_array)
            run_end = time.time()

        accuracy_array = np.array(accuracy_list)
        end = time.time()
        if store:
            result = {'time': run_end - run_start}
            result['acc_array'] = accuracy_array
            result['class_order'] = data_continuum.data_object.task_labels
            results.append(result)

    if store:
        save_file = open(save_stat_path, 'wb')
        pickle.dump(results, save_file)
        save_file.close()

    if params.online:
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    else:
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print("avg_end_acc {}".format(np.mean(accuracy_list)))
