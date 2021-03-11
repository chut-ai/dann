from dann.transfer.get_loader import get_mnist, get_svhn
from dann.basic_dann.models import Discriminator, Encoder, Classifier
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable

batch_size = 64

source_train, source_val = get_svhn(batch_size)
target_train, target_val = get_mnist(batch_size)


class DANN():
    def __init__(self):

        self.encoder = Encoder().cuda()
        self.classifier = Classifier().cuda()
        self.discriminator = Discriminator().cuda()

        parameters = list(self.encoder.parameters(
        )) + list(self.classifier.parameters()) + list(self.discriminator.parameters())

        self.optimizer = optim.Adam(parameters, 1e-3)

        self.loss_class = torch.nn.NLLLoss().cuda()
        self.loss_domain = torch.nn.NLLLoss().cuda()

    def train(self, n_epoch, source_train, target_train):
        """Source_train and target_train must have the same dimension"""

        for epoch in range(n_epoch):

            correct_source = 0
            correct_target = 0

            for i, (source_data, target_data) in enumerate(zip(source_train, target_train)):

                if i == len(source_train)-1:
                    break

                p = float(i + epoch*len(source_data)) / \
                    (n_epoch * len(source_data))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # Training NN using source data

                source_inputs, source_labels = source_data
                source_inputs = source_inputs.cuda()
                source_labels = source_labels.cuda()

                target_inputs, target_labels = target_data
                # Target labels just for evaluating model
                target_inputs = target_inputs.cuda()
                target_labels = target_labels.cuda()

                self.optimizer.zero_grad()
                batch_size = len(source_labels)

                source_domain_label = torch.zeros(batch_size).long().cuda()

                encoded = self.encoder(source_inputs)
                source_class_output = self.classifier(encoded)
                source_domain_output = self.discriminator(encoded, alpha)

                err_source_label = self.loss_class(
                    source_class_output, source_labels)
                err_source_domain = self.loss_domain(
                    source_domain_output, source_domain_label)

                # Training model using target data

                target_domain_label = torch.ones(batch_size).long().cuda()

                encoded = self.encoder(target_inputs)
                target_class_output = self.classifier(encoded)
                target_domain_output = self.discriminator(encoded, alpha)

                err_target_domain = self.loss_domain(
                    target_domain_output, target_domain_label)
                err = err_target_domain + err_source_label + err_source_domain
                # err = err_source_label
                err.backward()
                self.optimizer.step()

                # Get predicted values for source
                predicted_source = torch.max(source_class_output, 1)[1]
                correct_source += (predicted_source == source_labels).sum()
                correct_percent_source = correct_source*100/(batch_size*(i+1))

                # Get predicted values for target
                predicted_target = torch.max(target_class_output, 1)[1]
                correct_target += (predicted_target == target_labels).sum()
                correct_percent_target = correct_target*100/(batch_size*(i+1))

                if i % 50 == 0:
                    message = "Epochs : {}/{}, ({:.0f}%), Source accuracy:{:.3f}, Target accuracy:{:.3f}, Err source domain:{:.3f}, Err target domain:{:.3f}".format(
                        epoch+1, n_epoch, 100*i/len(source_train), correct_percent_source, correct_percent_target, err_source_domain, err_target_domain)
                    print(message)


dann = DANN()
dann.train(15, source_train, target_train)


# Evaluate model

dann.encoder.eval()
dann.classifier.eval()

with torch.no_grad():
    correct_source = 0
    for val_source_imgs, val_source_labels in source_val:
        val_source_imgs = Variable(val_source_imgs).float().cuda()
        outputs = dann.classifier(dann.encoder(val_source_imgs))
        predicted = torch.max(outputs, 1)[1].cuda()
        val_source_labels = val_source_labels.cuda()
        correct_source += (predicted == val_source_labels).sum()
    correct_source_percent = 100*correct_source/(len(source_val)*batch_size)
    print(correct_source_percent)
    correct_target = 0
    for val_target_imgs, val_target_labels in target_val:
        val_target_imgs = Variable(val_target_imgs).float().cuda()
        outputs = dann.classifier(dann.encoder(val_target_imgs))
        predicted = torch.max(outputs, 1)[1].cuda()
        val_target_labels = val_target_labels.cuda()
        correct_target += (predicted == val_target_labels).sum()
    correct_target_percent = 100*correct_target/(len(target_val)*batch_size)
    print(correct_target_percent)
