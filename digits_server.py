import numpy as np
import scipy.io as sio
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import digits_model
from utils import NormalizeRangeTanh, UnNormalizeRangeTanh


class ZeroPadBottom(object):
    ''' Zero pads batch of image tensor Variables on bottom to given size. Input (B, C, H, W) - padded on H axis. '''

    def __init__(self, size, use_gpu=True):
        self.size = size
        self.use_gpu = use_gpu

    def __call__(self, sample):
        B, C, H, W = sample.size()
        diff = self.size - H
        padding = Variable(torch.zeros(B, C, diff, W), requires_grad=False)
        if self.use_gpu:
            padding = padding.cuda()
        zero_padded = torch.cat((sample, padding), dim=2)
        return zero_padded


if torch.cuda.is_available():
    f_old_model = torch.load('./pretrained_model/model_F_SVHN_NormRange.tar')['best_model']
else:
    f_old_model = torch.load('./pretrained_model/model_F_SVHN_NormRange.tar', map_location='cpu')['best_model']

f_old_dict = f_old_model.state_dict()
f_new_model = digits_model.F(3, False)
f_new_dict = f_new_model.state_dict()
f_new_dict = {k: v for k, v in f_old_dict.items() if k in f_new_dict}
f_old_dict.update(f_new_dict)
f_new_model.load_state_dict(f_new_dict)
f_model = f_new_model

for param in f_model.parameters():
    param.requires_grad = False
f_model = f_model.eval()

if torch.cuda.is_available():
    model = torch.load('./final_models/fin_model.tar')['best_model']
else:
    model = torch.load('./final_models/fin_model.tar', map_location='cpu')['best_model']

loaded_mat = sio.loadmat('./data/svhn/test_32x32.mat')


def get_svhn_image(ind):
    data = loaded_mat['X']
    labels = loaded_mat['y'].astype(np.int64).squeeze()
    np.place(labels, labels == 10, 0)
    data = np.transpose(data, (3, 2, 0, 1))
    img0 = data[ind]
    img0 = Image.fromarray(np.transpose(img0, (1, 2, 0)))
    return img0


load_transforms = transforms.Compose([transforms.ToTensor(), NormalizeRangeTanh()])
unnormRange = UnNormalizeRangeTanh()


def digits_predict(img):
    image = load_transforms(img)
    image = image.unsqueeze(0)
    img_data = Variable(image.cpu().float())
    s_f = f_model(img_data)
    s_g = model['G'].cpu()(s_f)
    out_img = torchvision.utils.make_grid(unnormRange(s_g[:16]), nrow=4)

    npimg = out_img.detach().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    zero_array = np.zeros(npimg.shape)
    one_array = np.ones(npimg.shape)
    npimg = np.minimum(npimg, one_array)
    npimg = np.maximum(npimg, zero_array)

    return npimg
