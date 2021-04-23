import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from models.get_model import get_model
from dataset import load_data
from torch.utils.data import DataLoader
from config import *

class TestModel(unittest.TestCase):
    def run_model_common(self, dataset, dataset_file_path, model_name, dim_predict):
        data_train, data_test = load_data(dataset, dataset_file_path, model_name)
        model = get_model(model_name, dataset, rand_seed=seed, step_size=step_size, device=device, flatten_weight=False)
        data_loader = DataLoader(data_train, batch_size=1, shuffle=False, num_workers=0)
        for i, (images, labels) in enumerate(data_loader):
            w = model.get_weight()
            model.assign_weight(w)
            model.model.train()
            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()
            output = model.model(images)
            self.assertEqual(len(output[0]), dim_predict)
            loss = model.loss_fn(output, labels)
            loss.backward()
            model.optimizer.step()
            print(loss.item())
            break

    def test_mnist_run_model(self):
        self.run_model_common('MNIST', dataset_file_path, 'ModelCNNMnist', 10)
        self.run_model_common('MNIST', dataset_file_path, 'LeNet5', 10)

    def test_svhn_run_model(self):
        self.run_model_common('SVHN', dataset_file_path, 'WResNet40-2', 10)
        self.run_model_common('SVHN', dataset_file_path, 'WResNet16-1', 10)

    def test_cifar_10_run_model(self):
        self.run_model_common('cifar10', dataset_file_path, 'ModelCNNCifar10', 10)
        self.run_model_common('cifar10', dataset_file_path, 'ResNet34', 10)
        self.run_model_common('cifar10', dataset_file_path, 'ResNet18', 10)

    def test_cifar_100_run_model(self):
        self.run_model_common('cifar100', dataset_file_path, 'ResNet34', 100)
        self.run_model_common('cifar100', dataset_file_path, 'ResNet18', 100)

    def test_femnist_run_model(self):
        self.run_model_common('FEMNIST', dataset_file_path,  'ModelCNNEmnist', 62)
        self.run_model_common('FEMNIST', dataset_file_path, 'ModelCNNEmnistLeaf', 62)

    def test_celeba_run_model(self):
        self.run_model_common('celeba', dataset_file_path,  'ModelCNNCeleba', 2)

    def test_shakespeare_run_model(self):
        self.run_model_common('shakespeare', dataset_file_path, 'ModelLSTMShakespeare', 80)


if __name__ == '__main__':
    unittest.main()