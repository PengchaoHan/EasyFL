### Easy-to-Use Federated Learning Simulator in Pytorch (EasyFL)



#### Getting Started

The code runs on Python 3. To install the dependencies, run
```
pip3 install -r requirements.txt
```

For MNIST, SVHN, CIFAR-10, and CIFAR-100, the datasets will be downloaded automatically by the torchvision package.

For FEMNIST, manually download the dataset as instructed in <https://github.com/TalwalkarLab/leaf> and put the train and test datasets extended by .json into `dataset_data_files/femnist/train` and `dataset_data_files/femnist/test` respectively.

For CelebA, manually download the celebrity faces dataset img_align_celeba.zip from <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>, extract the files, and place the standalone .jpg files into `dataset_data_files/celeba/raw/img_align_celeba`.
Then, manually download the train and test files extended by .json as instructed by <https://github.com/TalwalkarLab/leaf> and put them into `dataset_data_files/celeba/train` and `dataset_data_files/celeba/test` respectively.

For Shakespeare, manually download the dataset as instructed in <https://github.com/TalwalkarLab/leaf> and put the train and test datasets extended by .json into `dataset_data_files/shakespeare/train` and `dataset_data_files/shakespeare/test` respectively.

To test the code: 

```
python simulation.py
```

#### Code Structure

All configuration options are given in `config.py` which also explains the different setups that the code can run with.

The results are saved as CSV files in the `results` folder. 
The CSV files should be deleted before starting a new round of experiment.
Otherwise, the new results will be appended to the existing file.

### Third-Party Library

Part of this code is inspired by 
<https://github.com/IBM/adaptive-federated-learning>,

<https://github.com/jz9888/federated-learning>,

 <https://github.com/bolianchen/Data-Free-Learning-of-Student-Networks>, 
 
 and <https://github.com/xternalz/WideResNet-pytorch>.


### Contributing
Any suggestion or issue is welcome. Pull request is very welcome.
