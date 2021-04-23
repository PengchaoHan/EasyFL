import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from dataset import load_data
from config import dataset_file_path
from torch.utils.data import DataLoader


class TestDataset(unittest.TestCase):
    def load_data_common(self, dataset, dataset_file_path, model_name, num_samples_to_read_train, num_samples_to_read_test,
                         dim_x, dim_y, dim_channel):
        data_train, data_test = load_data(dataset, dataset_file_path, model_name)

        if dataset == 'MNIST' or dataset == 'SVHN' or dataset == 'cifar10'or dataset == 'cifar100':
            self.assertEqual(len(data_train), num_samples_to_read_train)
            self.assertEqual(len(data_test), num_samples_to_read_test)
        data_loader = DataLoader(data_train, batch_size=1, shuffle=False, num_workers=0)
        if dataset == 'shakespeare':
            for i, (images, labels) in enumerate(data_loader):
                self.assertEqual(len(images[0]), dim_x)
                break
        else:
            for i, (images, labels) in enumerate(data_loader):
                self.assertEqual(len(images[0]), dim_channel)
                self.assertEqual(len(images[0][0]), dim_x)
                self.assertEqual(len(images[0][0][0]), dim_y)
                break

    def test_mnist_load_data(self):
        self.load_data_common('MNIST', dataset_file_path, 'ModelCNNMnist', 60000, 10000, 28, 28, 1)
        self.load_data_common('MNIST', dataset_file_path, 'LeNet5', 60000, 10000, 32, 32, 1)

    def test_svhn_load_data(self):
        self.load_data_common('SVHN', dataset_file_path, 'WResNet40-2', 73257, 26032, 32, 32, 3)

    def test_cifar_10_load_data(self):
        self.load_data_common('cifar10', dataset_file_path, 'ModelCNNCifar10', 50000, 10000, 32, 32, 3)

    def test_cifar_100_load_data(self):
        self.load_data_common('cifar100', dataset_file_path, 'ResNet34', 50000, 10000, 32, 32, 3)

    def test_femnist_load_data(self):
        self.load_data_common('FEMNIST', dataset_file_path,  'ModelCNNEmnist', -1, -1, 28, 28, 1)

    def test_celeba_load_data(self):
        self.load_data_common('celeba', dataset_file_path,  'ModelCNNCeleba', -1, -1, 84, 84, 3)

    def test_shakespeare_load_data(self):
        self.load_data_common('shakespeare', dataset_file_path,  'ModelLSTMShakespeare', -1, -1, 80, -1, -1)

if __name__ == '__main__':
    unittest.main()
