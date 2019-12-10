import numpy as np
import torch
from torchvision import datasets, transforms
import os
from utils.utils import template, img_conv2d
from PIL import Image
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading

def change_schedule(now_schedule,all_schedule):
    canvas.coords(fill_rec, (0, 0, 10 + 435 * (now_schedule/all_schedule), 20))
    top.update()


def draw_kernels(start, end, ns, kernel_size):
    global image 
    global tkImage
    global labels
    labels = [tk.Label(top) for _ in range(ns)]
    global images
    images = []

    Label_visker = tk.Label(top, text="Used Gaussian Kernels:")
    Label_visker.place(x=10, y=130)
    for s in range(ns):
        temp = template(kernel_size, start + s * (end - start) / ns)
        temp = temp / temp.max() * 255
        image = Image.fromarray(temp).resize((90, 90))
        tkImage = ImageTk.PhotoImage(image=image)
        images.append(tkImage)
        # image.show()
    for s in range(ns):
        labels[s].config(image=images[s])
        labels[s].place(x=10 + s * 90, y=160)

def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def draw_eample_results(ns):
    global result_image 
    global result_tkImage
    global result_labels
    result_labels = [tk.Label(top) for _ in range(ns)]
    global result_images
    result_images = []
    Label_visresult = tk.Label(top, text="Example Results", font=ft)
    Label_visresult.place(x=10, y=350)
    datasetname = comboxlist.get()
    for s in range(ns):
        if datasetname == 'MNIST':
            result_image = Image.open('data/augemented_data/{}/0/1-{}.bmp'.format(datasetname, s)).resize((90, 90))
        if datasetname == 'CIFAR10':
            result_image = Image.open('data/augemented_data/{}/6/0-{}.bmp'.format(datasetname, s)).resize((90, 90))
        result_tkImage = ImageTk.PhotoImage(image=result_image)
        result_images.append(result_tkImage)
    for s in range(ns):
        result_labels[s].config(image=result_images[s])
        result_labels[s].place(x=10 + s * 90, y=370)

# 1. make augemented data
def load():
    load_button.config(state='disabled')
    run_button.config(state='disabled')
    global train_loader
    datasetname = comboxlist.get()
    if datasetname == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ])),
            batch_size=1, shuffle=False)
    if datasetname == 'CIFAR10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ])),
            batch_size=1, shuffle=False)
    run_button.config(state='normal')



def run():
    run_button.config('disabled')
    start = int(s1.get())
    end = int(s2.get())
    ns = int(s3.get())
    kernel_size = int(s4.get())

    draw_kernels(start, end, ns, kernel_size)

    Progress = tk.Label(top, text="Running Progress", font=ft)
    Progress.place(x=10, y=260)

    global canvas
    canvas = tk.Canvas(top, width=465, height=22)
    canvas.place(x=10, y=280)

    global out_rec
    global fill_rec
    out_rec = canvas.create_rectangle(2,2,440,20,outline = "blue",width = 1)
    fill_rec = canvas.create_rectangle(2,2,0,20,outline = "",width = 0,fill = "green")

    datasetname = comboxlist.get()

    for idx, (data, target) in enumerate(train_loader):
        if datasetname == 'MNIST':
            data = data.numpy()[0][0] * 255
        if datasetname == 'CIFAR10':
            data = data.numpy()[0] *255
            data = data.transpose((1,2,0))
        target = target.numpy()[0]

        root = 'data/augemented_data/{}/{}'.format(datasetname, target)
        if os.path.exists(root) is False:
            os.makedirs(root)

        for s in range(ns):
            temp = template(kernel_size, start + s * (end - start) / ns)
            img = img_conv2d(data, temp)
            file_name = 'data/augemented_data/{}/{}/{}-{}.bmp'.format(datasetname, target, idx, s)
            if datasetname == 'MNIST':
                im = Image.fromarray(np.uint8(img)).convert('L')
            else:
                im = Image.fromarray(np.uint8(img))
            im.save(file_name)

        if idx % int(len(train_loader) / 600) == 0:
            change_schedule(idx, len(train_loader))
        if idx > 10:
            break

    Finish_label = tk.Label(top, wraplength = 520, justify = 'left', text="Done! The augmentated images are stored in \"data/augmente\nd_data/{}\" for next use in step2!".format(datasetname), font=ft)
    Finish_label.place(x=10, y=310)

    draw_eample_results(ns)
    run_button.config(state='normal')
    load_button.config(state='normal')

    

top = tk.Tk()
top.title('Multi-scale Defense of Adversarial Images')
ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
top.geometry('480x480')
top.resizable(width=False, height=False)
Label = tk.Label(top, text="Step1: Using Gaussian kernels to augment training images:", font=ft, anchor = 'w')
Label.place(x=10, y=0)

select_data = tk.Label(top, text="Select Dataset:")
select_data.place(x=10, y=30)

comvalue=tk.StringVar()#窗体自带的文本，新建一个值
comboxlist=ttk.Combobox(top,textvariable=comvalue) #初始化
comboxlist["values"]=("MNIST", "CIFAR10")
comboxlist.current(0)  #选择第一个
comboxlist.place(x=110, y=30)

Label1 = tk.Label(top, text="Scale Start:")
Label1.place(x=10, y=70)
s1 = tk.Entry(top, width=5)
s1.place(x=90, y=70)
s1.insert(tk.INSERT, "0")

Label2 = tk.Label(top, text="Scale End:")
Label2.place(x=10, y=100)
s2 = tk.Entry(top, width=5)
s2.place(x=90, y=100)
s2.insert(tk.INSERT, "2")

Label3 = tk.Label(top, text="Scale Number:")
Label3.place(x=150, y=70)
s3 = tk.Entry(top, width=5)
s3.place(x=250, y=70)
s3.insert(tk.INSERT, "5")
s3.config(state='disabled')

Label4 = tk.Label(top, text="Kernel Radius:")
Label4.place(x=150, y=100)
s4 = tk.Entry(top, width=5)
s4.place(x=250, y=100)
s4.insert(tk.INSERT, "2")

load_button = tk.Button(top, text='Load', command=lambda:thread_it(load), height=2, width=15)
load_button.place(x=350, y=30)

run_button = tk.Button(top, text='Run', command=lambda:thread_it(run), height=2,width=15)
run_button.place(x=350, y=80)
run_button.config(state='disabled')

top.mainloop()