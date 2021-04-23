import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import collections
from functools import reduce
from models.resnet import *
from models.lenet import *
from models.wresnet import *
from torch.autograd import Variable
import copy

LOSS_ACC_BATCH_SIZE = 128   # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class Models():
    def __init__(self, rand_seed=None, learning_rate=0.001, num_classes=10, model_name='LeNet5', channels=1, img_size=32, device=torch.device('cuda'), flatten_weight=False):
        super(Models, self).__init__()
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.model = None
        self.loss_fn = None
        self.weights_key_list = None
        self.weights_size_list = None
        self.weights_num_list = None
        self.optimizer = None
        self.channels = channels
        self.img_size = img_size
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate

        if model_name == 'ModelCNNMnist':
            from models.cnn_mnist import ModelCNNMnist
            self.model = ModelCNNMnist().to(device)
            self.init_variables()
        elif model_name == 'ModelCNNEmnist':
            from models.cnn_emnist import ModelCNNEmnist
            self.model = ModelCNNEmnist().to(device)
            self.init_variables()
        elif model_name == 'ModelCNNEmnistLeaf':
            from models.cnn_emnist_leaf import ModelCNNEmnist
            self.model = ModelCNNEmnist().to(device)
        elif model_name == 'ModelCNNCifar10':
            from models.cnn_cifar10 import ModelCNNCifar10
            self.model = ModelCNNCifar10().to(device)
            self.init_variables()
        elif model_name == 'ModelCNNCeleba':
            from models.cnn_celeba import ModelCNNCeleba
            self.model = ModelCNNCeleba().to(device)
        elif model_name == 'ModelLSTMShakespeare':
            from models.lstm_shakespeare import ModelLSTMShakespeare
            self.model = ModelLSTMShakespeare().to(device)
        elif model_name == 'LeNet5':
            self.model = LeNet5()
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # lr 0.001
        elif model_name == 'ResNet34':
            # import torchvision.models as models
            # self.model = models.resnet34(pretrained=False)  # outpu_feature=1000
            self.model = ResNet34(num_classes=num_classes)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # lr 0.1 adjustable
        elif model_name == 'ResNet18':
            # import torchvision.models as models
            # self.model = models.resnet18(pretrained=False)  # outpu_feature=1000
            self.model = ResNet18(num_classes=num_classes)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        elif model_name == 'WResNet40-2':
            self.model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif model_name == 'WResNet16-1':
            self.model = WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropRate=0.0)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def weight_variable(self, tensor, mean, std):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

    def bias_variable(self, shape):
        return torch.ones(shape) * 0.1

    def init_variables(self):

        self._get_weight_info()

        weight_dic = collections.OrderedDict()

        for i in range(len(self.weights_key_list)):
            if i%2 == 0:
                tensor = torch.zeros(self.weights_size_list[i])
                sub_weight = self.weight_variable(tensor, 0, 0.1)
            else:
                sub_weight = self.bias_variable(self.weights_size_list[i])
            weight_dic[self.weights_key_list[i]] = sub_weight

        self.model.load_state_dict(weight_dic)

    def _get_weight_info(self):
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w=0
            self.weights_num_list.append(num_w)

    def get_weight_dimension(self):
        dim = sum(self.weights_num_list)
        return dim

    def get_weight(self):
        with torch.no_grad():
            state = self.model.state_dict()
            if self.flatten_weight:
                weight_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(state[self.weights_key_list[0]].device)
                start_index = 0
                for i,[_, v] in zip(range(len(self.weights_num_list)), state.items()):
                    weight_flatten_tensor[start_index:start_index+self.weights_num_list[i]] = v.view(1, -1)
                    start_index += self.weights_num_list[i]

                return weight_flatten_tensor
            else:
                return copy.deepcopy(state)

    def assign_weight(self, w):
        if self.flatten_weight:
            self.assign_flattened_weight(w)
        else:
            self.model.load_state_dict(w)

    def assign_flattened_weight(self, w):

        weight_dic = collections.OrderedDict()
        start_index = 0

        for i in range(len(self.weights_key_list)):
            sub_weight = w[start_index:start_index+self.weights_num_list[i]]
            if len(sub_weight) > 0:
                weight_dic[self.weights_key_list[i]] = sub_weight.view(self.weights_size_list[i])
            else:
                weight_dic[self.weights_key_list[i]] = torch.tensor(0)
            start_index += self.weights_num_list[i]
        self.model.load_state_dict(weight_dic)

    def _data_reshape(self, imgs, labels=None):
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                _, y_label = torch.max(labels.data, 1)  # From one-hot to number
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    def gradient(self, imgs, labels, w, sampleIndices, device):
        self.assign_weight(w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        imgs = imgs[sampleIndices].to(device)
        labels = labels[sampleIndices].to(device)

        imgs, labels = self._data_reshape(imgs, labels)

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(imgs)
        loss = self.loss_fn(output, labels)
        loss.backward()

        return self.get_weight()

    def accuracy(self, data_test_loader, w, device):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)
        acc = float(total_correct) / len(data_test_loader.dataset)

        return avg_loss.item(), acc

    def predict(self, img, w, device):

        self.assign_weight(w)
        img, _ = self._data_reshape(img)
        with torch.no_grad():
            self.model.eval()
            _, pred = torch.max(self.model(img.to(device)).data, 1)

        return pred

    def train_one_epoch(self, data_train_loader, device):
        self.model.train()
        for i, (images, labels) in enumerate(data_train_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
