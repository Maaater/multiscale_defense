import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from mnist.mnist_net_F import Net
from mnist.utils import img_conv2d, template
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="batch classify images")
parser.add_argument('--image', type=str, default='attacks/iFGSM_ms/hacked_data/10-3-2.npy')
args = parser.parse_args()


print("\nStep4.3: Drawing confidence evolution curve of an image.\n")

# load model
net = Net()
params = torch.load('ckpt/model_augemented.pkl')
net.load_state_dict(params)
if torch.cuda.is_available():
    net.cuda()
net.eval()

trans = transforms.Compose([transforms.ToTensor()])

filename = args.image

img = np.load(filename)
C = np.zeros((20, 10))

for i in range(20):
    #  do gaussian blur
    temp = template(2, i * 0.1)
    input_image = img_conv2d(img, temp)
    input_image = np.expand_dims(input_image, 2)
    input_image = trans(input_image / 255)
    input_image = input_image.view(-1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image = input_image.float()
    preds = F.softmax(net(input_image.cuda()), dim=1)
    C[i, :] = preds.cpu().detach().numpy()

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

plt.figure(1, figsize=[6, 2])

ax1 = plt.subplot(1, 2, 1)
plt.imshow(Image.fromarray(img))
plt.xticks([])
plt.yticks([])
ax2 = plt.subplot(1, 2, 2)
plt.plot(f1[0:len(f1):1], '-r', linewidth=1)
plt.plot(f2[0:len(f2):1], '-b', linewidth=1)
plt.xticks([0, 10, 20])
plt.yticks([0,1])
plt.xlabel('Scale')
plt.ylabel('Confidence')

imname = filename.split('/')[-1][:-4]
plt.savefig("{dir}/{name}{form}".format(dir="evolution_curves", name=imname, form='.pdf'), dpi=300)
plt.show()

print("Done! The confidence evolution curve is saved as {dir}/{name}{form}.".format(dir="evolution_curves", name=imname, form='.pdf'))