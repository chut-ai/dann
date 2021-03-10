import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




net = CNN()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 5

for epoch in range(epochs):  # loop over the dataset multiple times

    correct = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

       # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # get precicted values
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum()
        correct_percent = correct*100/(batch_size*(i+1))

        if i % 50 == 0:
            message = "Epochs : {}/{}, ({:.0f}%), Loss:{:.6f}, Accuracy:{:.3f}".format(
                epoch+1, epochs, 100*i/len(trainloader), loss, correct_percent)
            print(message)

print('Finished Training')


correct = 0
for val_imgs, val_labels in valloader:
    val_imgs = Variable(val_imgs).float()
    outputs = net(val_imgs)
    predicted = torch.max(outputs, 1)[1]
    correct += (predicted == val_labels).sum()

correct_percent = 100*correct/(len(valloader)*batch_size)
print("Validation accuracy:{:.3f}% ".format(correct_percent))
