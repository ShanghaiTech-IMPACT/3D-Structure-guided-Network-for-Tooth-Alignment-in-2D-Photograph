import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from Stage1.SegmentToothContour.Model import UNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])

def SegmentToothContour(mouth, state, if_visual=True):
    ### build model
    model = UNet(n_classes=2)
    model.load_state_dict(torch.load(state))
    model.to(torch.device('cuda'))
    model.eval()

    ### initialize data
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)    # numpy_RGB_uint8
    mouth = transform(mouth)
    mouth = mouth.unsqueeze(0).cuda()
    
    with torch.no_grad():
        pred = model(mouth)
        pred = pred[0].cpu().numpy().argmax(0)
        pred = np.uint8(pred*255)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'result_ToothContour.png'), pred)
    return pred          #numpy_BGR_uint8

