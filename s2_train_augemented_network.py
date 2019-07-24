import numpy as np
import torch
from torchvision import datasets, transforms
import os
from mnist.mnist_net_F import Net
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F

print("Step2: Train a network for classifying handwriting digits with generated multi-scale image dataset:")
print("Strat training ...")

# train network
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('data/augemented_data',
                        transform=transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                   ])),
    batch_size=2000, num_workers=0 ,shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1000, shuffle=False)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# print("cuda is is_available:", torch.cuda.is_available())
use_cuda = torch.cuda.is_available()

torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
    model.cuda()

train_L = []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            train_L.append(loss.data.item())

test_acc = []

def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for test_data, test_target in test_loader:
        if use_cuda:
            test_data, test_target = test_data.cuda(), test_target.cuda()
        test_output = model(test_data)
        # sum up batch loss
        test_loss += F.cross_entropy(test_output, test_target, reduction='sum').item()
        # get the index of the max log-probability
        pred = test_output.data.max(1, keepdim=True)[1]
        correct += pred.eq(test_target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset)))

for epoch in range(10):
    train(epoch)
    test(test_loader)

torch.save(model.cpu().state_dict(), 'ckpt/model_augemented.pkl')
print("Done! The parameters of the trained network are stored as ckpt/model_augemented.pkl for next use in step3!")