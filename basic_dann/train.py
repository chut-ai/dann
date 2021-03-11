from dann.transfer.get_loader import get_mnist, get_svhn
from dann.basic_dann.models import Discriminator, Encoder, Classifier
import torch.optim as optim
import torch

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

            for i, (source_data, target_data) in enumerate(zip(source_train, target_train)):

                # Training NN using source data

                source_inputs, source_labels = source_data
                source_inputs = source_inputs.cuda()
                source_labels = source_labels.cuda()

                target_inputs, _ = target_data
                target_inputs = target_inputs.cuda()

                self.optimizer.zero_grad()
                batch_size = len(source_labels)

                domain_label = torch.zeros(batch_size).long().cuda()

                encoded = self.encoder(source_inputs)
                class_output = self.classifier(encoded)
                domain_output = self.discriminator(encoded)

                err_source_label = self.loss_class(class_output, source_labels)
                err_source_domain = self.loss_domain(
                    domain_output, domain_label)

                # Training model using target data

                domain_label = torch.ones(batch_size).long().cuda()

                encoded = self.encoder(target_inputs)
                domain_output = self.discriminator(encoded)

                err_target_domain = self.loss_domain(
                    domain_output, domain_label)
                err = err_target_domain + err_source_label + err_source_domain
                err.backward()
                self.optimizer.step()

                if i % 50 == 0:
                    message = "Epochs : {}/{}, ({:.0f}%), Loss:{:.6f}".format(
                        epoch+1, n_epoch, 100*i/len(source_train), err)
                    print(message)


dann = DANN()
dann.train(1, source_train, target_train)
