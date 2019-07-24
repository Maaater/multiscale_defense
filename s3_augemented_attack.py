import argparse
import foolbox
import torch
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification
import numpy as np
from torchvision import datasets, transforms
from mnist.mnist_net_F import Net
from PIL import Image
# from torchvision.models import vgg16
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch MSRCSAN')
parser.add_argument('--method', default='FGSM',
                        help='model name')
opt = parser.parse_args(["--method", "DeepFool"])
method = opt.method

print("\nStep3: Performing Attack:")

use_cuda = torch.cuda.is_available()
params = torch.load('ckpt/model_augemented.pkl')
torch_model = Net()
torch_model.load_state_dict(params)
if use_cuda:
    torch_model = torch.nn.DataParallel(torch_model).cuda()

torch_model.eval()
fmodel = PyTorchModel(torch_model, bounds=[0, 1], num_classes=10)

print(method, 'attacking:')
if method is 'FGSM':
    from foolbox.attacks import FGSM
    attack = FGSM(model=fmodel, criterion=Misclassification())
if method is 'iFGSM':
    from foolbox.attacks import IterativeGradientSignAttack
    attack = IterativeGradientSignAttack(model=fmodel, criterion=Misclassification())
if method is 'DeepFool':
    from foolbox.attacks import DeepFoolAttack
    attack = DeepFoolAttack(model=fmodel, criterion=Misclassification())

if os.path.exists('attacks/' + method + '_ms') is False:
    os.mkdir('attacks/' + method + '_ms')
    os.mkdir('attacks/' + method + '_ms' + '/hacked')
    os.mkdir('attacks/' + method + '_ms' + '/hacked_data')
    os.mkdir('attacks/' + method + '_ms' + '/original')
    os.mkdir('attacks/' + method + '_ms' + '/original_data')           
hacked_path = 'attacks/' + method + '_ms' + '/hacked'    
hacked_data_path = 'attacks/' + method + '_ms' + '/hacked_data'
original_path = 'attacks/' + method + '_ms' + '/original'
original_data_path = 'attacks/' + method + '_ms' + '/original_data'


# Load test images
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                        ])),
    batch_size=1, shuffle=True)

count = 1
for data, label in test_loader:
    data = data[0].numpy()
    label = label.item()
    adversarial = attack(data, label)
    
    if adversarial is not None:    
        if np.linalg.norm(adversarial - data) == 0:
            # print("no adversarial images found, hack next image")
            continue
        adv_label = np.argmax(fmodel.predictions(adversarial))
        # print(adv_label)
        # print(foolbox.utils.softmax(fmodel.predictions(adversarial))[adv_label])

        if np.linalg.norm(adversarial - data) > 0:
            image_data = np.zeros([28, 28])
            image_data = adversarial[0] * 255

            hackedname = hacked_data_path + '/' + str(count) + '-' + str(label) + '-' + str(adv_label) + ".npy"
            np.save(hackedname, image_data)
            image = Image.fromarray(image_data.astype(np.uint8))
            image.save("{hackedpath}/{name}-{label}-{adv_label}.png".format(hackedpath=hacked_path,name=count, label=label, adv_label=adv_label))

            ori_data = np.zeros([28,28])
            ori_data = data[0] * 255

            oriname = original_data_path + '/' + str(count) + '-' + str(label) + ".npy"
            np.save(oriname, ori_data)

            oriimage = Image.fromarray(ori_data.astype(np.uint8))
            oriimage.save("{originalpath}/{name}-{label}.png".format(originalpath=original_path,name=count,label=label))
            count = count + 1
        
        if count % 50 == 0:
            print("Attack: {}/{}".format(count, 500))
        if count > 500:
            break

print("Done! The adversarial images and correspoinding data are stored in attacks for next use in step4!")