# Code adapted from https://github.com/bolianchen/Data-Free-Learning-of-Student-Networks

#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feature=False, out_activation=False):
        activation1 = self.conv1(img)
        output = self.relu1(activation1)
        output = self.maxpool1(output)
        activation2 = self.conv2(output)
        output = self.relu2(activation2)
        output = self.maxpool2(output)
        activation3 = self.conv3(output)
        output = self.relu3(activation3)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            if out_activation == True:
                return output, feature, activation1, activation2, activation3
            else:
                return output, feature


class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feature=False, out_activation=False):  # [batch_size, 1, 32, 32]
        activation1 = self.conv1(img)  # [batch_size, 3, 28, 28]
        output = self.relu1(activation1)  # [batch_size, 3, 28, 28]
        output = self.maxpool1(output)  # [batch_size, 3, 14, 14]
        activation2 = self.conv2(output)  # [batch_size, 8, 10, 10]
        output = self.relu2(activation2)  # [batch_size, 8, 10, 10]
        output = self.maxpool2(output)  # [batch_size, 8, 5, 5]
        activation3 = self.conv3(output)  # [batch_size, 60, 1, 1]
        output = self.relu3(activation3)  # [batch_size, 60, 1, 1]
        feature = output.view(-1, 60)  # [batch_size, 60]
        output = self.fc1(feature)  # [batch_size, 42]
        output = self.relu4(output)
        output = self.fc2(output)  # [batch_size, 10]
        if out_feature == False:
            return output
        else:
            if out_activation == True:
                return output, feature, activation1, activation2, activation3
            else:
                return output, feature
