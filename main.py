import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import data_participation_each_node
import numpy as np
import random
from model.model import Model
from util.util import DatasetSplit, WorkerSampler
import copy

if device.type != 'cpu':
    torch.cuda.set_device(device)


if __name__ == "__main__":
    stat = CollectStatistics(results_eval_file_prefix=results_file_prefix)

    if save_agg_weight:
        agg_weight_filename = 'aggregationWeights_' + results_file_prefix + '.csv'
        with open(agg_weight_filename, 'a') as f:
            f.write('sim_seed,num_iter,aggregation_weight_each_node\n')
        participation_count_filename = 'participationCount_' + results_file_prefix + '.csv'
        with open(participation_count_filename, 'a') as f:
            f.write('sim_seed,num_iter,participation_count_each_node\n')

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
        dict_users, participation_prob_each_node = data_participation_each_node(data_train, n_nodes)

        step_size_local = step_size_local_config
        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)

        train_loader_list = []
        dataiter_list = []
        for n in range(n_nodes):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = next(dataiter_list[n])
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = next(dataiter_list[n])
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])

            return images, labels

        def sample_full_batch(n):
            images = []
            labels = []
            for i in range(len(train_loader_list[n].dataset)):
                images.append(train_loader_list[n].dataset[i][0])

                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)

        w_global = model.get_weight()   # Get initial weight

        num_iter = 0
        last_output = 0
        last_save_latest = 0
        last_save_checkpoint = 0

        not_participate_count_at_node = []
        participate_intervals_at_node = []
        worker_samplers = []
        if save_agg_weight:
            agg_weight_each_node = []
            participation_count_each_node = []
        for n in range(n_nodes):
            not_participate_count_at_node.append(0)
            participate_intervals_at_node.append([])
            worker_samplers.append(WorkerSampler(participation_method, participation_prob_each_node[n]))
            if save_agg_weight:
                agg_weight_each_node.append(None)
                participation_count_each_node.append(0)

        if weighting_method == 'fedvarp' or weighting_method == 'mifa':
            update_per_node = []
            for n in range(n_nodes):
                update_per_node.append(torch.zeros(w_global.shape[0]).to('cpu'))
            update_per_node = torch.stack(update_per_node)
        if weighting_method == 'fedvarp':
            update_all_avg = torch.zeros(w_global.shape[0]).to('cpu')

        learning_rate_decay_level = 0

        while True:
            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))
                with open(participation_count_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))

            if num_iter > lr_decay_start_iter:
                if int((num_iter - lr_decay_start_iter) / lr_decay_interval) + 1 > learning_rate_decay_level and \
                        learning_rate_decay_level < 10:
                    learning_rate_decay_level += 1
                    step_size_local /= 2
                    model.update_learning_rate(step_size_local)

            print('seed', seed, ' - iteration', num_iter, ' - local step size', step_size_local)
            accumulated = 0

            for n in range(n_nodes):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participated = True

                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, iters_per_round):
                        images, labels = sample_minibatch(n)

                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous()  # contiguous() needed due to the use of ColorJitter in CIFAR transforms

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()
                        model.optimizer.step()

                    w_tmp = model.get_weight()  # deepcopy is already included here
                    w_tmp -= w_global  # This is the difference (i.e., update) in this round

                    agg_weight = None
                    if weighting_method == 'known_prob':
                        w_tmp /= worker_sampler.participation_prob
                        agg_weight = 1/worker_sampler.participation_prob
                    elif weighting_method == 'adaptive':
                        if len(participate_intervals_at_node[n]) > 0:
                            agg_weight = np.mean(participate_intervals_at_node[n])
                            w_tmp *= agg_weight
                    elif weighting_method == 'fedvarp':
                        w_tmp_new = copy.deepcopy(w_tmp).to('cpu')
                        w_tmp -= update_per_node[n].to(device)
                        update_per_node[n] = w_tmp_new
                    elif weighting_method == 'mifa':
                        update_per_node[n] = w_tmp.to('cpu')

                    if save_agg_weight:
                        agg_weight_each_node[n] = agg_weight
                        participation_count_each_node[n] += 1

                    participate_intervals_at_node[n].append(not_participate_count_at_node[n] + 1)  # interval used in the next round
                    not_participate_count_at_node[n] = 0

                else:
                    participated = False
                    not_participate_count_at_node[n] += 1

                    if not_participate_count_at_node[n] >= k_value:
                        participate_intervals_at_node[n].append(not_participate_count_at_node[n])
                        not_participate_count_at_node[n] = 0

                    w_tmp = 0.0

                if save_agg_weight:
                    with open(agg_weight_filename, 'a') as f:
                        if agg_weight_each_node[n] is not None:
                            f.write(',' + "%.4f" % agg_weight_each_node[n])
                        else:
                            f.write(',nan')
                    with open(participation_count_filename, 'a') as f:
                        f.write(',' + str(participation_count_each_node[n]))

                if weighting_method != 'mifa':  # No need to accumulate for MIFA
                    if accumulated == 0:  # accumulated weights
                        w_accumulate = w_tmp
                        # Note: w_tmp cannot be used after this
                    else:
                        w_accumulate += w_tmp

                    if weighting_method != 'average_participating' and weighting_method != 'fedvarp':
                        accumulated += 1
                    elif participated:
                        accumulated += 1

            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write('\n')
                with open(participation_count_filename, 'a') as f:
                    f.write('\n')

            if weighting_method != 'mifa':
                if accumulated > 0:
                    w_tmp_a = torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
                    if weighting_method == 'fedvarp':
                        w_tmp_a += update_all_avg.to(device)
                        update_all_avg = torch.mean(update_per_node, 0)
                else:
                    w_tmp_a = torch.zeros(w_global.shape[0]).to(device)
            else:
                w_tmp_a = torch.mean(update_per_node, 0).to(device)

            w_global += torch.tensor(step_size_global).to(device) * w_tmp_a

            num_iter = num_iter + iters_per_round

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                stat.collect_stat_eval(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                last_output = num_iter

            if num_iter >= max_iter:
                break

        if save_final_model:
            torch.save(model.model.state_dict(), 'final_model_sim' + str(seed) + '_iter' + str(num_iter) + '_' + save_model_file)

        del model
        del w_global
        if weighting_method != 'mifa':
            del w_accumulate

        torch.cuda.empty_cache()
