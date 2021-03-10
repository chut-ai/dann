import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dann.transfer import model, get_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64

mnist_train, mnist_val = get_loader.get_mnist(batch_size)
svhn_train, svhn_val = get_loader.get_svhn(batch_size)

net = model.CNN()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 15

for epoch in range(epochs):  # loop over the dataset multiple times

    correct = 0

    for i, data in enumerate(svhn_train, 0):
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
                epoch+1, epochs, 100*i/len(svhn_train), loss, correct_percent)
            print(message)

print('Finished Training')


correct_percent_mnist = net.evaluate(mnist_val, batch_size)
correct_percent_svhn = net.evaluate(svhn_val, batch_size)

print("Validation accuracy on MNIST:{:.3f}% ".format(correct_percent_mnist))
print("Validation accuracy on SVHN:{:.3f}% ".format(correct_percent_svhn))
