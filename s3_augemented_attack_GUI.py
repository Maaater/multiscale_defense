import foolbox
import torch
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")
import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading

from models.LeNet import Net
from models.ResNet import resnet32


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def send_information(receiver, text):
    receiver.config(state='normal')
    receiver.insert("end", text)
    receiver.config(state='disabled')

def select_model():
    network = net_comboxlist.get()
    ckpt = checkpoint.get()
    log_text = 'Selected Model: {}\n'.format(network)
    send_information(Receive_window, log_text)
        
    if network == 'LeNet':
        model = Net()
        params = torch.load(ckpt)
        model.load_state_dict(params)
    if network == 'ResNet32':
        checkp = torch.load(ckpt)
        model = checkp['net']
    return model

def load_data():
    dataset = comboxlist.get()
    log_text = 'Select Dataset: {}\n'.format(dataset)
    send_information(Receive_window, log_text)

    if dataset == 'MNIST':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ])),
            batch_size=1, shuffle=True)
    if dataset == 'CIFAR10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ])),
            batch_size=1, shuffle=True)  
    return test_loader

def run():
    run_button.config(state='disabled')
    test_loader = load_data()
    torch_model = select_model()
    if use_cuda.get():
        log_text = 'Move model to cuda.'
        torch_model = torch_model.cuda()

    torch_model.eval()
    fmodel = PyTorchModel(torch_model, bounds=[0, 1], num_classes=10)

    method = attack_comboxlist.get()
    log_text = 'Perform {} attack... \n'.format(method)
    send_information(Receive_window, log_text)
    if method == 'FGSM':
        from foolbox.attacks import FGSM
        attack = FGSM(model=fmodel, criterion=Misclassification())
    if method == 'iFGSM':
        from foolbox.attacks import IterativeGradientSignAttack
        attack = IterativeGradientSignAttack(model=fmodel, criterion=Misclassification())
    if method == 'DeepFool':
        from foolbox.attacks import DeepFoolAttack
        attack = DeepFoolAttack(model=fmodel, criterion=Misclassification())


    attack_root = 'attacks/' + comboxlist.get() + '/' + method + '_ms'
    hacked_path = attack_root + '/hacked'
    hacked_data_path = attack_root + '/hacked_data'
    original_path = attack_root + '/original'
    original_data_path = attack_root + '/original_data'
    if os.path.exists(attack_root) is False:
        os.makedirs(attack_root)
        os.mkdir(hacked_path)
        os.mkdir(hacked_data_path)
        os.mkdir(original_path)
        os.mkdir(original_data_path)           

    count = 1
    for data, label in test_loader:
        data = data[0].numpy()
        label = label.item()
        adversarial = attack(data, label)
        
        if adversarial is not None:    
            if np.linalg.norm(adversarial - data) == 0:
                continue
            adv_label = np.argmax(fmodel.predictions(adversarial))

            if np.linalg.norm(adversarial - data) > 0:
                if save_adv.get():
                    # image_data = np.zeros([28, 28])

                    # TODO : needs to be fixed
                    image_data = adversarial[0] * 255
                    hackedname = hacked_data_path + '/' + str(count) + '-' + str(label) + '-' + str(adv_label) + ".npy"
                    np.save(hackedname, image_data)
                    image = Image.fromarray(image_data.astype(np.uint8))
                    image.save("{hackedpath}/{name}-{label}-{adv_label}.png".format(hackedpath=hacked_path,name=count, label=label, adv_label=adv_label))

                if save_ori.get():
                    # ori_data = np.zeros([28,28])
                    ori_data = data[0] * 255

                    oriname = original_data_path + '/' + str(count) + '-' + str(label) + ".npy"
                    np.save(oriname, ori_data)

                    oriimage = Image.fromarray(ori_data.astype(np.uint8))
                    oriimage.save("{originalpath}/{name}-{label}.png".format(originalpath=original_path,name=count,label=label))
                
                count = count + 1
            
            if count % (att_num.get() / 10) == 0:
                log_text = "Attack: {}/{}".format(count, att_num.get())
                send_information(Receive_window, log_text)
            if count > att_num.get():
                break

    log_text = "Done! The adversarial images and correspoinding data are stored in attacks for next use in step4!"
    send_information(Receive_window, log_text)
    run_button.config(state='normal')


top = tk.Tk()
top.title('Multi-scale Defense of Adversarial Images')
ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
top.geometry('480x430')
top.resizable(width=False, height=False)
Label = tk.Label(top, text="Step3: Performing Attack:", font=ft, anchor = 'w')
Label.place(x=10, y=10)

select_data = tk.Label(top, text="Select Dataset:")
select_data.place(x=10, y=40)
comvalue=tk.StringVar()#窗体自带的文本，新建一个值
comboxlist=ttk.Combobox(top,textvariable=comvalue) #初始化
comboxlist["values"]=("MNIST", "CIFAR10")
comboxlist.current(0)  #选择第一个
comboxlist.place(x=130, y=40)

select_network = tk.Label(top, text="Select Network:")
select_network.place(x=10, y=70)
net_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
net_comboxlist=ttk.Combobox(top,textvariable=net_comvalue) #初始化
net_comboxlist["values"]=("LeNet", "ResNet32")
net_comboxlist.current(0)  #选择第一个
net_comboxlist.place(x=130, y=70)

select_checkpoint = tk.Label(top, text="Select Checkpoint:")
select_checkpoint.place(x=10, y=100)
checkpoint = tk.Entry(top, width=23)
checkpoint.place(x=130, y=100)
checkpoint.insert(tk.INSERT, "ckpt/LeNet_augmented.pkl")

select_attack = tk.Label(top, text="Select Attack:")
select_attack.place(x=10, y=130)
attack_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
attack_comboxlist=ttk.Combobox(top,textvariable=attack_comvalue) #初始化
attack_comboxlist["values"]=("FGSM", "iFGSM", "DeepFool")
attack_comboxlist.current(0)  #选择第一个
attack_comboxlist.place(x=130, y=130)

attack_number = tk.Label(top, text="Attack number:")
attack_number.place(x=10, y=160)
att_num = tk.Entry(top, width=23)
att_num.place(x=130, y=160)
att_num.insert(tk.INSERT, "500")

use_cuda = tk.IntVar()            #用来表示按钮是否选中
c1 = tk.Checkbutton(top,text='Use CUDA',variable=use_cuda)
c1.place(x=350, y=40)

save_adv = tk.IntVar()            #用来表示按钮是否选中
s_adv = tk.Checkbutton(top,text='Save Adversarial Images',variable=save_adv)
s_adv.place(x=10, y=190)

save_ori = tk.IntVar()            #用来表示按钮是否选中
s_ori = tk.Checkbutton(top,text='Save Original Image',variable=save_ori)
s_ori.place(x=190, y=190)

Receive = tk.LabelFrame(top, text='Running information',padx=10, pady=10)
Receive.place(x=10, y=220)
Receive_window = scrolledtext.ScrolledText(Receive, width=57, height=10, padx=10, pady=10, wrap=tk.WORD)
Receive_window.grid()

run_button = tk.Button(top, text='Perform\nAttack', command=lambda:thread_it(run), height=5,width=15)
run_button.place(x=350, y=70)

top.mainloop()