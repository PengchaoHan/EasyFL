import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from datasets.dataset import get_data_info


def get_model(model_name, dataset, rand_seed=None, step_size=None, device=torch.device('cuda'), flatten_weight=False):
    img_size, channels, num_classes = get_data_info(dataset, model_name)

    from models.model import Models
    return Models(rand_seed, step_size, num_classes=num_classes, model_name=model_name, channels=channels,
                  img_size=img_size, device=device, flatten_weight=flatten_weight)


def adjust_learning_rate(optimizer, n_iter, learning_rate):
    lr = learning_rate / torch.sqrt(torch.tensor(n_iter, dtype=torch.float32))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
