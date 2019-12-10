import numpy as np
import torch
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


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def send_information(receiver, text):
    receiver.config(state='normal')
    receiver.insert("end", text)
    receiver.config(state='disabled')

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
    
    for s in range(5):
        labels[s].config(image=images[s])
        labels[s].place(x=10 + s * 90, y=235)


def draw_preds_lines(preds):
    global line_label
    
    figure = plt.figure(figsize=(9,4))
    ax = figure.add_subplot(111)

    for i in range(10):
        x = [0,1,2,3,4,5,6,7,8,9]
        ax.plot(x, np.array(preds[:, i]))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])  #去掉y轴
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
    plt.xticks(size=15)
    plt.tight_layout()
    plt.savefig("temp/predline.png")
    plt.close

    image = Image.open("temp/predline.png").resize((440, 200))
    global tkImage
    tkImage = ImageTk.PhotoImage(image=image)

    # image.show()
    line_label.config(image=tkImage)
    line_label.place(x=500, y=20)

def draw_roc_curves(C, filename, img):
    l1 = np.argmax(C[0, :])

    c = np.zeros(10)

    for i in range(10):
        c[i] = max(C[:, i]) - min(C[:, i])

    top_index = c.argsort()[-2:][::-1]

    if l1 == top_index[0]:
        f1 = C[:, l1]
        f2 = C[:, top_index[1]]
    else:
        f1 = C[:, l1]
        f2 = C[:, top_index[0]]

    fontsize = 20
    font = {'family':'Calibri','size': fontsize}
    plt.figure(1, figsize=[6, 3])

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(f1[0:len(f1):1], '-r', linewidth=1)
    plt.plot(f2[0:len(f2):1], '-b', linewidth=1)
    plt.xticks([0,1,2,3,4,5,6,7,8,9], fontsize=fontsize)
    plt.yticks([0,1], fontsize=fontsize)
    plt.xlabel('Scale', fontdict=font)
    plt.ylabel('Confidence', fontdict=font)

    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Calibri') for label in labels]

    imname = filename.split('/')[-1][:-4]
    plt.tight_layout()
    if os.path.exists("evolution_curves") is False:
        os.mkdir("evolution_curves")
    plt.savefig("{dir}/{name}{form}".format(dir="evolution_curves", name=imname, form='.pdf'), dpi=300)
    plt.close()


def run(data_source):
    total_num = 100
    run_once_button.config(state='disabled')
    run_all_button.config(state='disabled')

    start = float(s1.get())
    end = float(s2.get())
    ns = int(s3.get())
    kernel_size = int(s4.get())

    draw_kernels(start, end, ns, kernel_size)

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

    Acc = 0

    Count = 0 

    global line_label
    line_label = tk.Label(top)

    image_labels = [tk.Label(top) for _ in range(ns)]

    pathdir = imagedir.get() + data_source
    for name in os.listdir(pathdir):
        filename = os.path.join(pathdir, name)
        img = np.load(filename)
        Count += 1
        C = np.zeros((ns, 10))

        global exam_images
        exam_images = []

        for i in range(ns):
            #  do gaussian blur
            temp = template(2, i * (2/ns))
            input_image = img_conv2d(img, temp)

            if i < 5:
                image = Image.fromarray(np.uint8(input_image)).resize((90, 90))
                exam_images.append(ImageTk.PhotoImage(image=image))
                image_labels[i].config(image=exam_images[i])
                image_labels[i].place(x=10 + i * 90, y=360)

            if len(input_image.shape)==2:
                input_image = np.expand_dims(input_image, 2)
            input_image = trans(input_image / 255)
            input_image = input_image.view(-1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
            input_image = input_image.float()

            if use_cuda.get():
                input_image = input_image.cuda()

            preds = F.softmax(net(input_image), dim=1)
            C[i, :] = preds.cpu().detach().numpy()

            draw_preds_lines(C)
            
        draw_roc_curves(C, name, Image.fromarray(np.uint8(img)))
        
        c = np.zeros(10)

        for i in range(10):
            c[i] = max(C[:, i]) - min(C[:, i])

        fluc = max(c)

        t = 0.5
        if fluc > t :
            Acc += 1

        if Count >= total_num:
            acc = Acc / total_num
            log_text = "Done! The attack detection accuracy is {.2f}!".format(acc)
            send_information(Receive_window, log_text)
            break
        time.sleep(2)
        
    run_once_button.config(state='normal')
    run_all_button.config(state='normal')


top = tk.Tk()
top.title('Multi-scale Defense of Adversarial Images')
ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
top.geometry('960x470')
top.resizable(width=False, height=False)
Label = tk.Label(top, text="Step4: Conducting multiscale detection on adversarial images", font=ft, anchor = 'w')
Label.place(x=10, y=10)

# Select network button
select_network = tk.Label(top, text="Select Network:")
select_network.place(x=10, y=50)
net_comvalue=tk.StringVar()#窗体自带的文本，新建一个值
net_comboxlist=ttk.Combobox(top,textvariable=net_comvalue, width=25) #初始化
net_comboxlist["values"]=("LeNet", "ResNet32","Vgg16")
net_comboxlist.current(1)  #选择第一个
net_comboxlist.place(x=130, y=50)

select_checkpoint = tk.Label(top, text="Select Checkpoint:")
select_checkpoint.place(x=10, y=80)
checkpoint = tk.Entry(top, width=28)
checkpoint.place(x=130, y=80)
checkpoint.insert(tk.INSERT, "ckpt/ckpt32_ms.pkl")

Image_floder = tk.Label(top, text="Image Folder:")
Image_floder.place(x=10, y=110)
imagedir = tk.Entry(top, width=28)
imagedir.place(x=130, y=110)
imagedir.insert(tk.INSERT, "attacks/CIFAR10/FGSM_ms/")

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
s3.insert(tk.INSERT, "10")
s3.config(state="disabled")

Label4 = tk.Label(top, text="Kernel Radius:")
Label4.place(x=150, y=180)
s4 = tk.Entry(top, width=5)
s4.place(x=250, y=180)
s4.insert(tk.INSERT, "2")


Receive = tk.LabelFrame(top, text='Running information',padx=10, pady=10)
Receive.place(x=490, y=250)
Receive_window = scrolledtext.ScrolledText(Receive, width=57, height=10, padx=10, pady=10, wrap=tk.WORD)
Receive_window.grid()

run_once_button = tk.Button(top, text='Run Normal', command=lambda:thread_it(run, "original_data"), height=3,width=15)
run_once_button.place(x=350, y=70)
run_all_button = tk.Button(top, text='Run Attacked', command=lambda:thread_it(run, "hacked_data"), height=3,width=15)
run_all_button.place(x=350, y=140)

top.mainloop()