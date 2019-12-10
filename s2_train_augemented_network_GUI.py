import numpy as np
import torch
from torchvision import datasets, transforms
import os
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading

from models.LeNet import Net
from models.ResNet import resnet32

def send_information(receiver, text):
    receiver.config(state='normal')
    receiver.insert("end", text)
    receiver.config(state='disabled')


# train network
def load_data():
    dataset = comboxlist.get()
    log_text = 'Select Dataset: {}\n'.format(dataset)
    send_information(Receive_window, log_text)

    if dataset == 'Augmented_MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('data/augemented_data/MNIST',
                                transform=transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                        ])),
            batch_size=500, num_workers=0 ,shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=1000, shuffle=False)

    if dataset == 'Augmented_CIFAR10':
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('data/augemented_data/CIFAR10',
                                transform=transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                        ])),
            batch_size=500, num_workers=0 ,shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=1000, shuffle=False)     
    
    return train_loader, test_loader


def select_model():
    network = net_comboxlist.get()
    log_text = 'Selected Model: {}\n'.format(network)
    send_information(Receive_window, log_text)
    if network == 'LeNet':
        model = Net()
    if network == 'ResNet32':
        model = resnet32()
    return model


def select_Optimizer(model):
    opt = optim_comboxlist.get()
    log_text = 'Selected Optimizer: {}\n'.format(opt)
    send_information(Receive_window, log_text)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer


def train(epoch, model, optimizer, use_cuda, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda.get():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            log_text = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item())
            send_information(Receive_window, log_text)
            

def test(model, use_cuda, test_loader):
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
    log_text = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset))
    send_information(Receive_window,log_text)


def run():
    run_button.config(state='disabled')
    train_data, test_data = load_data()
    model = select_model()
    optimizer = select_Optimizer(model)

    if use_cuda.get():
        torch.cuda.manual_seed(1)
        model.cuda()
    else:
        torch.manual_seed(1)

    for epoch in range(int(epoch_num.get())):
        train(epoch, model, optimizer, use_cuda, train_data)
        test(model, use_cuda, test_data)

    network = net_comboxlist.get()
    if network == 'LeNet':
        torch.save(model.cpu().state_dict(), 'ckpt/LeNet_augmented.pkl')
    if network == 'ResNet32':
        torch.save(model.cpu().state_dict(), 'ckpt/ResNet32_augmented.pkl')
    
    log_text = 'Done! The parameters of the trained network are stored as ckpt/{}_augmented.pkl for next use in step3!'.format(network)
    send_information(Receive_window, log_text)
    run_button.config(state='normal')


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


top = tk.Tk()
top.title('Multi-scale Defense of Adversarial Images')
ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
top.geometry('480x430')
top.resizable(width=False, height=False)
Label = tk.Label(top, text="Step2: Train a network for classifying handwriting digits\nwith generated multi-scale image dataset:", font=ft, anchor = 'w')
Label.place(x=10, y=0)

select_data = tk.Label(top, text="Select Dataset:")
select_data.place(x=10, y=50)
comvalue=tk.StringVar()#窗体自带的文本，新建一个值
comboxlist=ttk.Combobox(top,textvariable=comvalue) #初始化
comboxlist["values"]=("Augmented_MNIST", "Augmented_CIFAR10")
comboxlist.current(0)  #选择第一个
comboxlist.place(x=120, y=50)

select_network = tk.Label(top, text="Select Network:")
select_network.place(x=10, y=80)
net_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
net_comboxlist=ttk.Combobox(top,textvariable=net_comvalue) #初始化
net_comboxlist["values"]=("LeNet", "ResNet32")
net_comboxlist.current(0)  #选择第一个
net_comboxlist.place(x=120, y=80)

select_optimizer = tk.Label(top, text="Select Optimizer:")
select_optimizer.place(x=10, y=110)
optim_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
optim_comboxlist=ttk.Combobox(top,textvariable=optim_comvalue) #初始化
optim_comboxlist["values"]=("SGD", "Adam")
optim_comboxlist.current(1)  #选择第一个
optim_comboxlist.place(x=120, y=110)

select_loss = tk.Label(top, text="Select Loss:")
select_loss.place(x=10, y=140)
loss_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
loss_comboxlist=ttk.Combobox(top,textvariable=loss_comvalue) #初始化
loss_comboxlist["values"]=("Cross_entropy", "Others")
loss_comboxlist.current(0)  #选择第一个
loss = comboxlist.get()
loss_comboxlist.place(x=120, y=140)

epoch = tk.Label(top, text="Set Epochs:")
epoch.place(x=10, y=170)
epoch_num = tk.Entry(top, width=22)
epoch_num.place(x=120, y=170)
epoch_num.insert(tk.INSERT, "10")

use_cuda = tk.IntVar()            #用来表示按钮是否选中
c1 = tk.Checkbutton(top,text='Use CUDA',variable=use_cuda)
c1.place(x=350, y=40)

Receive = tk.LabelFrame(top, text='Running information',padx=10, pady=10)
Receive.place(x=10, y=200)
Receive_window = scrolledtext.ScrolledText(Receive, width=57, height=12, padx=10, pady=10, wrap=tk.WORD)
Receive_window.grid()

run_button = tk.Button(top, text='Train', command=lambda:thread_it(run), height=5,width=15)
run_button.place(x=350, y=70)

top.mainloop()