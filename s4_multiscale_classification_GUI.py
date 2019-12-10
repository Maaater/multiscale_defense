import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import argparse
import os
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading

from models.LeNet import Net
from utils.utils import img_conv2d, template


def send_information(receiver, text):
    receiver.config(state='normal')
    receiver.insert("end", text)
    receiver.config(state='disabled')

def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def draw_kernels(start, end, ns, kernel_size):
    global image 
    #global tkImage
    global labels
    labels = [tk.Label(top) for _ in range(ns)]
    global images
    images = []

    Label_visker = tk.Label(top, text="Used Gaussian Kernels:")
    Label_visker.place(x=10, y=210)
    for s in range(ns):
        temp = template(kernel_size, start + s * (end - start) / ns)
        temp = temp / temp.max() * 255
        image = Image.fromarray(temp).resize((90, 90))
        tkImage = ImageTk.PhotoImage(image=image)
        images.append(tkImage)
        # image.show()
    
    for s in range(ns):
        labels[s].config(image=images[s])
        labels[s].place(x=10 + s * 90, y=235)


def draw_preds(preds, i):
    global bar_images
    global bar_labels
    x = [0,1,2,3,4,5,6,7,8,9]
    figure = plt.figure(figsize=(2,2))
    ax = figure.add_subplot(111)
    ax.bar(x, np.array(preds[0]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])  #去掉y轴
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
    plt.xticks(size=15)
    plt.tight_layout()
    if os.path.exists("temp/") is False:
        os.mkdir("temp")

    plt.savefig("temp/{}.png".format(str(i)))

    image = Image.open("temp/{}.png".format(str(i))).resize((90, 90))
    tkImage = ImageTk.PhotoImage(image=image)
    bar_images.append(tkImage)
    # image.show()
    bar_labels[i].config(image=bar_images[i])
    if i > 5-1:
        bar_labels[i].place(x=480 + 2 * 90, y=140)
    else:
        bar_labels[i].place(x=480 + i * 90, y=40)


def run(run_num):

    run_once_button.config(state='disabled')
    run_all_button.config(state='disabled')

    start = int(s1.get())
    end = int(s2.get())
    ns = int(s3.get())
    kernel_size = int(s4.get())

    draw_kernels(start, end, ns, kernel_size)

    # load model
    ckpt = checkpoint.get()
    if net_comboxlist.get() == 'LeNet':
        net = Net()
        params = torch.load(ckpt)
        net.load_state_dict(params)
        trans = transforms.Compose([transforms.ToTensor()])
    if net_comboxlist.get() == 'ResNet32':
        checkp = torch.load(ckpt, map_location='cpu')
        net = checkp['net']
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                 std=[0.2023, 0.1994, 0.2010])])
    

    if use_cuda.get():
        net.cuda()
    net.eval()

    C = 0 
    ns = 5
    count_succ = np.zeros(ns+1)

    Label_visexa = tk.Label(top, text="Evolved images:")
    Label_visexa.place(x=10, y=340)
    image_labels = [tk.Label(top) for _ in range(ns)]

    pathdir = imagedir.get()

    Label_vispic = tk.Label(top, text="Histogram of Evolved Predictions:")
    Label_vispic.place(x=480, y=10)

    global bar_labels
    bar_labels = [tk.Label(top) for _ in range(ns+1)]
    
    for name in os.listdir(pathdir):
        filename = os.path.join(pathdir, name)
        true_label = int(filename[-7])
        wrong_label = int(filename[-5])
        img = np.load(filename)
        C += 1
        scale_preds = np.zeros([1, 10])
        global exam_images
        exam_images = []

        global bar_images
        bar_images = []
        for i in range(ns):
            #  do gaussian blur
            temp = template(2, i * 0.4)
            input_image = img_conv2d(img, temp)
            # for visualization
            image = Image.fromarray(np.uint8(input_image)).resize((90, 90))
            
            exam_images.append(ImageTk.PhotoImage(image=image))
            image_labels[i].config(image=exam_images[i])
            image_labels[i].place(x=10 + i * 90, y=360)

            if net_comboxlist.get() == 'LeNet':
                input_image = np.expand_dims(input_image, 2)

            input_image = trans(input_image / 255)
            input_image = input_image.view(-1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
            input_image = input_image.float()

            if use_cuda.get():
                input_image = input_image.cuda()
            preds = F.softmax(net(input_image), dim=1)

            preds = preds.cpu().detach().numpy()
            scale_preds += preds

            draw_preds(preds, i)

            pred_label = np.argmax(preds)
            if pred_label == int(true_label):
                count_succ[i] += 1
        
        scale_preds = scale_preds / ns

        Label_viscon = tk.Label(top, text="Multiscale confidences:")
        Label_viscon.place(x=480, y=150)
        draw_preds(scale_preds, ns)
        scale_label = np.argmax(scale_preds)

        logtext = "Classification result:\n"
        send_information(Receive_window, logtext)
        logtext = "True label: {}\n".format(true_label)
        send_information(Receive_window, logtext)
        if scale_label == int(true_label):
            count_succ[ns] += 1
            logtext = "Before multiscale defense {}, after multiscale defense {}, defense successes!\n".format(wrong_label, scale_label)
            send_information(Receive_window, logtext)
        else:
            logtext = "Before multiscale defense {}, after multiscale defense {}, defense fails!\n".format(wrong_label, scale_label)
            send_information(Receive_window, logtext)

        if C % 50 == 0:
            log_text = "processing: {} / {}\n".format(C, run_num)
            send_information(Receive_window, log_text)
        if C >= run_num:
            break
        time.sleep(1)

    log_text = "Before defense: {} / {} \t  Classification accuracy:{:.2f} \n".format(0, run_num, 0)
    send_information(Receive_window, log_text)
    log_text = "After defense:  {} / {} \t Classification accuracy:{:.2f} \n".format(int(count_succ[ns]), run_num, count_succ[ns]/run_num)
    send_information(Receive_window, log_text)

    run_once_button.config(state='normal')
    run_all_button.config(state='normal')

top = tk.Tk()
top.title('Multi-scale Defense of Adversarial Images')
ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
top.geometry('960x470')
top.resizable(width=False, height=False)
Label = tk.Label(top, text="Step4: Conducting multiscale classification on adversarial\nimages", font=ft, anchor = 'w')
Label.place(x=10, y=10)

select_network = tk.Label(top, text="Select Network:")
select_network.place(x=10, y=50)
net_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
net_comboxlist=ttk.Combobox(top,textvariable=net_comvalue, width=25) #初始化
net_comboxlist["values"]=("LeNet", "ResNet32")
net_comboxlist.current(0)  #选择第一个
net_comboxlist.place(x=130, y=50)

select_checkpoint = tk.Label(top, text="Select Checkpoint:")
select_checkpoint.place(x=10, y=80)
checkpoint = tk.Entry(top, width=28)
checkpoint.place(x=130, y=80)
checkpoint.insert(tk.INSERT, "ckpt/model_augemented.pkl")

Image_floder = tk.Label(top, text="Image Folder:")
Image_floder.place(x=10, y=110)
imagedir = tk.Entry(top, width=28)
imagedir.place(x=130, y=110)
imagedir.insert(tk.INSERT, "attacks/MNIST/FGSM_ms/hacked_data")

use_cuda = tk.IntVar()            #用来表示按钮是否选中
c1 = tk.Checkbutton(top,text='Use CUDA',variable=use_cuda)
c1.place(x=350, y=40)


Label1 = tk.Label(top, text="Scale Start:")
Label1.place(x=10, y=150)
s1 = tk.Entry(top, width=5)
s1.place(x=90, y=150)
s1.insert(tk.INSERT, "0")

Label2 = tk.Label(top, text="Scale End:")
Label2.place(x=10, y=180)
s2 = tk.Entry(top, width=5)
s2.place(x=90, y=180)
s2.insert(tk.INSERT, "2")

Label3 = tk.Label(top, text="Scale Number:")
Label3.place(x=150, y=150)
s3 = tk.Entry(top, width=5)
s3.place(x=250, y=150)
s3.insert(tk.INSERT, "5")

Label4 = tk.Label(top, text="Kernel Radius:")
Label4.place(x=150, y=180)
s4 = tk.Entry(top, width=5)
s4.place(x=250, y=180)
s4.insert(tk.INSERT, "2")


Receive = tk.LabelFrame(top, text='Running information',padx=10, pady=10)
Receive.place(x=490, y=250)
Receive_window = scrolledtext.ScrolledText(Receive, width=57, height=10, padx=10, pady=10, wrap=tk.WORD)
Receive_window.grid()

run_once_button = tk.Button(top, text='Run Once', command=lambda:thread_it(run, 1), height=3,width=15)
run_once_button.place(x=350, y=70)
run_all_button = tk.Button(top, text='Run All', command=lambda:thread_it(run, 500), height=3,width=15)
run_all_button.place(x=350, y=140)

top.mainloop()