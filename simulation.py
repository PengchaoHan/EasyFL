# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning

from torch.utils.data import Dataset,DataLoader
from config import *
from datasets.dataset import *
from models.get_model import get_model, adjust_learning_rate
from statistic.collect_stat import CollectStatistics
from util.sampling import split_data
import numpy as np
import random
import copy

random.seed(seed)
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

data_train, data_test = load_data(dataset, dataset_file_path, model_name)
data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)  # num_workers=8
data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)  # num_workers=8
dict_users = split_data(dataset, data_train, n_nodes, iid)
if n_nodes is None:
    n_nodes = len(dict_users)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

model = get_model(model_name, dataset, rand_seed=seed, step_size=step_size, device=device, flatten_weight=flatten_weight)

stat = CollectStatistics(results_file_name=fl_results_file_path)
train_loader_list = []
dataiter_list = []
for n in range(n_nodes):
    train_loader_list.append(DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
    dataiter_list.append(iter(train_loader_list[n]))

w_global_init = model.get_weight()
w_global = copy.deepcopy(w_global_init)

num_iter = 0
last_output = 0

while True:
    w_global_prev = copy.deepcopy(w_global)
    if n_nodes_in_each_round > n_nodes:
        print("Warning: Not enough nodes for each round! set to all nodes.")
        n_nodes_in_each_round = n_nodes
    if random_node_selection:
        node_subset = np.random.choice(range(n_nodes), n_nodes_in_each_round, replace=False)
    else:
        node_subset = range(0, n_nodes_in_each_round)

    w_accu = None
    for n in node_subset:
        model.assign_weight(w_global)
        # model.train_one_epoch(train_loader_list[n], device)
        if dataset == 'shakespeare':
            adjust_learning_rate(model.optimizer, num_iter, step_size)
        model.model.train()
        for i in range(0, tau_setup):
            try:
                images, labels = dataiter_list[n].next()
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = dataiter_list[n].next()
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = dataiter_list[n].next()

            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()
            output = model.model(images)
            loss = model.loss_fn(output, labels)
            loss.backward()
            model.optimizer.step()

        w = model.get_weight()   # deepcopy is already included here

        if w_accu is None:  # accumulated weights
            w_accu = w
        else:
            if flatten_weight:
                w_accu += w
            else:
                for k in w_accu.keys():
                    w_accu[k] += w[k]

    num_iter = num_iter + tau_setup
    if num_iter >= max_iter:
        break

    if aggregation_method == 'FedAvg':
        if flatten_weight:
            w_global = torch.div(copy.deepcopy(w_accu), torch.tensor(n_nodes_in_each_round).to(device)).view(-1)
        else:
            for k in w_global.keys():
                w_global[k] = torch.div(copy.deepcopy(w_accu[k]), torch.tensor(n_nodes_in_each_round).to(device))
    else:
        raise Exception("Unknown parameter server method name")

    has_nan = False
    if flatten_weight:
        if (True in torch.isnan(w_global)) or (True in torch.isinf(w_global)):
            has_nan = True
    else:
        for k in w_global.keys():
            if (True in torch.isnan(w_global[k])) or (True in torch.isinf(w_global[k])):
                has_nan = True
    if has_nan:
        print('*** w_global is NaN or InF, using previous value')
        w_global = copy.deepcopy(w_global_prev)

    if num_iter - last_output >= num_iter_one_output:
        stat.collect_stat_global(num_iter, model, data_train_loader, data_test_loader, w_global)
        last_output = num_iter

    if num_iter >= max_iter:
        break

stat.collect_stat_end()
