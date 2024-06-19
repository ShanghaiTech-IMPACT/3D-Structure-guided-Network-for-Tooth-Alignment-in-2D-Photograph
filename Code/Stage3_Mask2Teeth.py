import yaml
import json
import cv2
import os
import torch
from Stage3.Network import Network
import numpy as np
from torchvision.utils import make_grid
import math
from PIL import Image


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()


def Stage3_Mask2Teeth(data, mode, state, if_visual=False):
    if mode in ['M2M2T']:
        from Stage3.Generator import Mask2TeethGenerator as Generator
        with open("./Stage3/config/config_Mask2Teeth.yaml", 'r') as f:
            GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
    elif mode in ['C2C2T_v1', 'C2C2T_v2']:
        from Stage3.Generator import Contour2TeethGenerator as Generator
        with open("./Stage3/config/config_Contour2Teeth.yaml", 'r') as f:
            GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
    elif mode in ['C2C2T_v2_facecolor_teethcolor']:
        from Stage3.Generator import Contour2ToothGenerator_FaceColor_TeethColor as Generator
        with open("./Stage3/config/config_Contour2Tooth_facecolor_teethcolor.yaml", 'r') as f:
            GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
    elif mode in ['C2C2T_v2_facecolor_lightcolor']:
        from Stage3.Generator import Contour2ToothGenerator_FaceColor_LightColor as Generator
        with open("./Stage3/config/config_Contour2Tooth_facecolor_lightcolor.yaml", 'r') as f:
            GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
    elif mode in ['C2C2T_v2_fourier']:
        from Stage3.Generator import Contour2ToothGenerator_Fourier as Generator
        with open("./Stage3/config/config_Contour2Tooth_Fourier.yaml", 'r') as f:
            GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']

    # initialize the Network
    netG = Network(GeneratorConfig['unet'], GeneratorConfig['beta_schedule'])
    # netG.load_state_dict(torch.load('Stage3/stage3_ckpt_8000.pth'), strict=False)
    netG.load_state_dict(torch.load(state), strict=False)
    netG.to(torch.device('cuda'))
    netG.eval()

    # initialize the Generator
    generator = Generator(netG)
    prediction, cond_teeth_color = generator.predict(data)       # tensor_BGR_float32 (-1to1)
    mouth_align = tensor2img(prediction)                          # numpy_BGR_uint8 (0-255)
    # cond_teeth_color = tensor2img(cond_teeth_color)              # numpy_BGR_uint8 (0-255)

    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'mouth_align.png'), mouth_align)

    return {
        "crop_mouth_align": mouth_align,              #numpy_BGR_uint8
        "cond_teeth_color": cond_teeth_color          #numpy_BGR_uint8
    }

