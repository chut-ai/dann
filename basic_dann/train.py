import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
from matplotlib.pyplot import imshow, imsave
from dann.transfer.get_loader import get_mnist, get_svhn
from dann.basic_dann.models import Encoder, Classifier, Discriminator
MODEL_NAME = 'DANN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

E = Encoder().to(DEVICE)
C = Classifier().to(DEVICE)
D = Discriminator().to(DEVICE)

batch_size = 64

tgt_train_loader, tgt_test_loader = get_mnist(batch_size)
src_train_loader, src_test_loader = get_svhn(batch_size)

domain_loss = nn.BCELoss()
class_loss = nn.CrossEntropyLoss()

E_opt = torch.optim.Adam(E.parameters())
C_opt = torch.optim.Adam(C.parameters())
D_opt = torch.optim.Adam(D.parameters())

n_epochs = 50
step = 0

D_src = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real
D_tgt = torch.zeros(batch_size, 1).to(DEVICE)  # Discriminator Label to fake
D_labels = torch.cat([D_src, D_tgt], dim=0)


def get_lambda(epoch, n_epochs):
    p = epoch / n_epochs
    return 2. / (1+np.exp(-10.*p)) - 1.



ll_c, ll_d = [], []
acc_lst = []

for epoch in range(1, n_epochs+1):
    for idx, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_images, src_labels = src_data
        tgt_images, _ = tgt_data
        # Training Discriminator
        src, labels, tgt = src_images.to(DEVICE), src_labels.to(
            DEVICE), tgt_images.to(DEVICE)

        x = torch.cat([src, tgt], dim=0)
        h = E(x)
        y = D(h.detach())

        Ld = domain_loss(y, D_labels)
        D.zero_grad()
        Ld.backward()
        D_opt.step()

        c = C(h[:batch_size])
        y = D(h)
        Lc = class_loss(c, labels)
        Ld = domain_loss(y, D_labels)
        lamda = 0.1*get_lambda(epoch, n_epochs)
        Ltot = Lc - lamda*Ld

        E.zero_grad()
        C.zero_grad()
        D.zero_grad()

        Ltot.backward()

        C_opt.step()
        E_opt.step()

        if step % 100 == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f} ---- {}'.format(
                epoch, n_epochs, step, Ld.item(), Lc.item(), lamda, dt))
            ll_c.append(Lc.detach().cpu())
            ll_d.append(Ld.detach().cpu())

        if step % 500 == 0:
            E.eval()
            C.eval()
            with torch.no_grad():
                corrects = torch.zeros(1).to(DEVICE)
                for idx, (src, labels) in enumerate(src_test_loader):
                    src, labels = src.to(DEVICE), labels.to(DEVICE)
                    c = C(E(src))
                    _, preds = torch.max(c, 1)
                    corrects += (preds == labels).sum()
                acc = corrects.item() / len(src_test_loader.dataset)
                print('***** Source acc: {:.3f}, Step: {}'.format(acc, step))

                corrects = torch.zeros(1).to(DEVICE)
                for idx, (tgt, labels) in enumerate(tgt_test_loader):
                    tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                    c = C(E(tgt))
                    _, preds = torch.max(c, 1)
                    corrects += (preds == labels).sum()
                acc = corrects.item() / len(tgt_test_loader.dataset)
                print('***** Target acc: {:.3f}, Step: {}'.format(acc, step))
                acc_lst.append(acc)

            E.train()
            C.train()
        step += 1

# Plots losses and accuracies

X = [100*step for step in range(len(ll_c))]

plt.figure(1)
plt.plot(X, ll_c)
plt.xlabel("number of batches")
plt.ylabel("label classification loss")

plt.figure(2)
plt.plot(X, ll_d)
plt.xlabel("number of batches")
plt.ylabel("domain classification loss")

X2 = [500*step for step in range(len(acc_lst))]

plt.figure(3)
plt.plot(X2, acc_lst)
plt.xlabel("number of batches")
plt.ylabel("target accuracy")

plt.show()
