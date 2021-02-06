from torchvision.datasets import VisionDataset
from PIL import Image
import os.path
import json
import numpy as np


class FEMNIST(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(FEMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        self.train_data_dir = os.path.join(root, 'train')
        self.test_data_dir = os.path.join(root, 'test')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_dir = self.train_data_dir
        else:
            data_dir = self.test_data_dir

        data_files = os.listdir(data_dir)

        data = {}

        data_files = [f for f in data_files if f.endswith('.json')]
        for f in data_files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            data.update(cdata['user_data'])

        list_keys = list(data.keys())
        self.images = []
        self.targets = []
        self.clients = []

        for i in range(0, len(list_keys)):
            # note: each time we append a list
            imgs = data[list_keys[i]]["x"]
            targets = data[list_keys[i]]["y"]

            for img in imgs:
                img = np.array(img, dtype='float32').reshape(28, 28)
                img = Image.fromarray(img, mode='F')
                if self.transform is not None:
                    img = self.transform(img)

                self.images.append(img)

            self.targets += targets

            for j in range(0, len(data[list_keys[i]]["x"])):
                self.clients.append(i)

    def __getitem__(self, index):
        img, target = self.images[index], int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_dict_clients(self):
        dict_clients = {}
        for i in range(0, len(self.clients)):
            if self.clients[i] not in dict_clients:
                dict_clients[self.clients[i]] = []
            dict_clients[self.clients[i]].append(i)

        for i in dict_clients.keys():
            dict_clients[i] = set(dict_clients[i])

        return dict_clients

    def download(self):
        raise Exception('Download currently not supported')

    def _check_exists(self):
        return os.path.exists(self.train_data_dir) and os.path.exists(self.test_data_dir)
