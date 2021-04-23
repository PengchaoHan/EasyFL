from torchvision.datasets import VisionDataset
import os.path
import json
from util.language_utils import process_x, process_y

class SHAKESPEARE(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(SHAKESPEARE, self).__init__(root, transform=transform,
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
        self.inputs = []
        self.targets = []
        self.clients = []

        for i in range(0, len(list_keys)):
            # note: each time we append a list
            inputs = data[list_keys[i]]["x"]
            targets = data[list_keys[i]]["y"]

            for input_ in inputs:
                input_ = process_x(input_)
                # input_ = np.array(input_, dtype='float32')
                if self.transform is not None:
                    input_ = self.transform(input_)
                self.inputs.append(input_.reshape(-1))

            for target in targets:
                target = process_y(target)
                self.targets += target[0]

            for j in range(0, len(data[list_keys[i]]["x"])):
                self.clients.append(i)

    def __getitem__(self, index):
        input1, target = self.inputs[index], int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input1, target

    def __len__(self):
        return len(self.inputs)

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
