import torch.nn as nn
import torch
from dann.basic_dann.function import ReverseLayerF
from torch.autograd import Variable


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 1, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def evaluate(self, source_val, target_val):
        self.eval()
        with torch.no_grad():
            correct_source = 0
            nb_elem = 0
            for source_data in source_val:
                source_inputs = Variable(source_data[0]).float().cuda()
                source_labels = Variable(source_data[1]).float().cuda()
                nb_elem += source_inputs.size()[0]
                source_output, _ = self(source_inputs, 1)
                predicted = torch.max(source_output, 1)[1].cuda()
                correct_source += (predicted == source_labels).sum()
            correct_source_percent = 100*correct_source/(nb_elem)

            correct_target = 0
            nb_elem = 0
            for target_data in target_val:
                target_inputs = Variable(target_data[0]).float().cuda()
                target_labels = Variable(target_data[1]).float().cuda()
                nb_elem += target_inputs.size()[0]
                source_output, _ = self(source_inputs, 1)
                target_output, _ = self(target_inputs, 1)
                predicted = torch.max(target_output, 1)[1].cuda()
                correct_target += (predicted == target_labels).sum()
            correct_target_percent = 100*correct_target/(nb_elem)
        self.train()
        return correct_source_percent, correct_target_percent


