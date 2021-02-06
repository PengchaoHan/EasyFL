import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from tensorboardX import SummaryWriter


class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv'):
        self.results_file_name = results_file_name
        self.summary_write = SummaryWriter(log_dir=os.path.join(results_file_path, comments))
        self.summary_write_train = SummaryWriter(log_dir=os.path.join(results_file_path, comments, 'train'))
        self.summary_write_test = SummaryWriter(log_dir=os.path.join(results_file_path, comments, 'test'))

        with open(results_file_name, 'a') as f:
            f.write(
                'num_iter,lossValue,trainAccuracy,predictionAccuracy\n')
            f.close()

    def collect_stat_global(self, num_iter, model, train_data_loader, test_data_loader, w_global=None):
        loss_value, train_accuracy = model.accuracy(train_data_loader, w_global, device)
        _, prediction_accuracy = model.accuracy(test_data_loader, w_global, device)

        self.summary_write.add_scalar('Loss', loss_value, num_iter)
        self.summary_write_train.add_scalar('Accuracy', train_accuracy, num_iter)
        self.summary_write_test.add_scalar('Accuracy', prediction_accuracy, num_iter)
        print("Iter. " + str(num_iter) + "  train accu " + str(train_accuracy) + "  test accu " + str(prediction_accuracy))

        with open(self.results_file_name, 'a') as f:
            f.write(str(num_iter) + ',' + str(loss_value) + ','
                    + str(train_accuracy) + ',' + str(prediction_accuracy) + '\n')
            f.close()

    def collect_stat_end(self):
        self.summary_write.close()
        self.summary_write_train.close()
        self.summary_write_test.close()

# tensorboard --logdir results
# tensorboard --logdir results --host=127.0.0.1

