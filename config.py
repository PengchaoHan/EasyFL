import os
import torch

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

# dataset = 'MNIST'
# model_name = 'ModelCNNMnist'
# model_name = 'LeNet5'

# dataset = 'cifar10'
# model_name = 'ModelCNNCifar10'
# model_name = 'ResNet34'
# model_name = 'ResNet18'

# dataset = 'cifar100'
# model_name = 'ResNet34'
# model_name = 'ResNet18'

# dataset = 'SVHN'
# model_name = 'WResNet40-2'
# model_name = 'WResNet16-1'

dataset = 'FEMNIST'
# model_name = 'ModelCNNEmnist'
model_name = 'ModelCNNEmnistLeaf'

# dataset = 'celeba'
# model_name = 'ModelCNNCeleba'

# dataset = 'shakespeare'
# model_name = 'ModelLSTMShakespeare'

dataset_file_path = os.path.join(os.path.dirname(__file__), 'dataset_data_files')
results_file_path = os.path.join(os.path.dirname(__file__), 'results/')
comments = dataset + "-" + model_name
fl_results_file_path = os.path.join(results_file_path, 'rst_' + comments + '.csv')

# ----------------------settings for clients
n_nodes = None  # None for fmnist, celeba, and shakespeare, set a number for others
n_nodes_in_each_round = 10
step_size = 0.01  # learning rate of clients, 0.8 for shakespeare, 0.01 for others
batch_size_train = 32
batch_size_eval = 256
max_iter = 100000  # Maximum number of iterations to run
seed = 1
aggregation_method = 'FedAvg'
random_node_selection = True
flatten_weight = False
# iid = True  # only for MNIST and CIFAR*
iid = False
tau_setup = 10  # number of iterations in local training
num_iter_one_output = 50
