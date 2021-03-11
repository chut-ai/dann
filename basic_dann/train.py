from dann.transfer.get_loader import get_mnist, get_svhn
from dann.basic_dann.models import CNNModel
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable

batch_size = 64
lr = 1e-3
n_epoch = 1

source_train, source_val = get_svhn(batch_size)
target_train, target_val = get_mnist(batch_size)

DANN = CNNModel().cuda()

optimizer = optim.Adam(DANN.parameters(), lr)

loss_class = torch.nn.NLLLoss().cuda()
loss_domain = torch.nn.NLLLoss().cuda()

for p in DANN.parameters():
    p.requires_grad = True

for epoch in range(n_epoch):

    len_dataloader = len(source_train)

    for i, (source_data, target_data) in enumerate(zip(source_train, target_train)):

        if len(source_data) != len(target_data):
            break

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
        batch_size = len(source_data)

        # Train model using source data

        domain_label = torch.zeros(batch_size).long().cuda()

        class_output, domain_output = DANN(source_inputs, alpha)
        err_s_label = loss_class(class_output, source_labels)
        err_s_domain = loss_domain(domain_output, domain_label)

        # Train model using target data

        domain_label = torch.ones(batch_size).long().cuda()

        _, domain_output = DANN(target_inputs, alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        err = err_t_domain + err_s_domain + err_s_label

        err.backward()
        optimizer.step()

        if i % 50 == 0:
            message = "Epochs : {}/{}, ({:.0f}%), Err source domain:{:.3f}, Err source label:{:.3f}, Err target domain:{:.3f}".format(
                epoch+1, n_epoch, 100*i/len(source_train), err_s_domain, err_s_label, err_t_domain)
            print(message)


"""
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
"""
