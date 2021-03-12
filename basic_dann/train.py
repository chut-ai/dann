from dann.transfer.get_loader import get_mnist, get_svhn
from dann.basic_dann.models import CNNModel
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

BATCH_SIZE = 64
lr = 1e-3
n_epoch = 15

source_train, source_val = get_svhn(BATCH_SIZE)
target_train, target_val = get_mnist(BATCH_SIZE)

DANN = CNNModel().cuda()

optimizer = optim.Adam(DANN.parameters(), lr)

loss_class = torch.nn.CrossEntropyLoss().cuda()
loss_domain = torch.nn.CrossEntropyLoss().cuda()

acc_source = []
acc_target = []

for p in DANN.parameters():
    p.requires_grad = True

for epoch in range(n_epoch):

    len_dataloader = len(source_train)

    for i, (source_data, target_data) in enumerate(zip(source_train, target_train)):

        p = float(i + epoch * len_dataloader) / (n_epoch * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_inputs, source_labels = source_data
        source_inputs = source_inputs.cuda()
        source_labels = source_labels.cuda()

        target_inputs, target_labels = target_data
        # Target labels just for evaluating model
        target_inputs = target_inputs.cuda()
        target_labels = target_labels.cuda()

        DANN.zero_grad()
        batch_size = len(source_inputs)

        # Train model using source data

        domain_label = torch.zeros(batch_size).long().cuda()

        class_output, domain_output = DANN(source_inputs, alpha)

        err_s_label = loss_class(class_output, source_labels)
        err_s_domain = loss_domain(domain_output, domain_label)

        # Train model using target data

        batch_size = len(target_inputs)

        domain_label = torch.ones(batch_size).long().cuda()

        _, domain_output = DANN(target_inputs, alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        err = err_t_domain + err_s_domain + err_s_label

        err.backward()
        optimizer.step()

        if i % 250 == 0:
            correct_source_percent, correct_target_percent = DANN.evaluate(
                source_val, target_val)

            message = "Epochs : {}/{}, ({:.0f}%), Source accuracy:{:.3f}%, Target accuracy:{:.3f}%".format(
                epoch+1, n_epoch, 100*i/len(source_train), correct_source_percent, correct_target_percent)
            print(message)

    correct_source_percent, correct_target_percent = DANN.evaluate(source_val, target_val)
    acc_source.append(correct_source_percent.cpu().numpy())
    acc_target.append(correct_target_percent.cpu().numpy())

X = range(1, n_epoch+1)

plt.figure()
plt.plot(X, acc_source, "r")
plt.plot(X, acc_target, "g")
plt.show()

