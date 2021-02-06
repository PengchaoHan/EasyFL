from torchvision.datasets import VisionDataset
from PIL import Image
import os.path
import json


class CelebA(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, read_all_data_to_mem=True):
        super(CelebA, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.read_all_data_to_mem = read_all_data_to_mem

        self.train_data_dir = os.path.join(root, 'train')
        self.test_data_dir = os.path.join(root, 'test')
        self.raw_data_dir = os.path.join(root, 'raw', 'img_align_celeba')

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download img_align_celeba.zip from '
                               'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and put it into dataset_files/celeba/raw. '
                               'Then, extract the files into the same folder. The standalone jpg files should be located '
                               'in dataset_files/celeba/raw/img_align_celeba after extraction.')

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
            img_files = data[list_keys[i]]["x"]
            if not self.read_all_data_to_mem:
                self.images += img_files
            else:
                for img_file in img_files:
                    img = Image.open(os.path.join(self.raw_data_dir, img_file))
                    # img.show()
                    if self.transform is not None:
                        img = self.transform(img)
                    self.images.append(img)

            self.targets += data[list_keys[i]]["y"]

            for j in range(0, len(data[list_keys[i]]["x"])):
                self.clients.append(i)

        print('')

    def __getitem__(self, index):
        img, target = self.images[index], int(self.targets[index])
        if not self.read_all_data_to_mem:
            img = Image.open(os.path.join(self.raw_data_dir, img))
            # img.show()
            if self.transform is not None:
                img = self.transform(img)

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
        raise Exception('Download currently not supported.')

    def _check_exists(self):
        return os.path.exists(self.train_data_dir) and os.path.exists(self.test_data_dir) \
               and os.path.exists(self.raw_data_dir)
