import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
import os

def Preprocess(img):
    # Input: cv2 image (BGR)
    img_copy = deepcopy(img)
    location = np.where((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0))
    img_copy[location[0], location[1]] = (255, 255, 255)
    # cv2.imwrite(r'C:\Users\douyl\Desktop\o3d\test.jpg',img_copy)
    return img_copy


def DetectContour(img, if_visual=True):
    # Input: cv2 image (BGR)
    # img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 预处理，背景变成白色
    img_preprocess = Preprocess(img)

    # img = cv2.GaussianBlur(img_preprocess, (3,3), 0)

    # Canny Edge Detection
    threshold1 = 200
    threshold2 = 100
    img_contour = cv2.Canny(img_preprocess, threshold1, threshold2)
    # 闭操作，闭合曲线
    # img_contour = cv2.morphologyEx(img_contour, cv2.MORPH_CLOSE, kernel=(3,3), iterations=2)
    # 膨胀
    kernel = np.ones((2,2), np.uint8)
    img_contour = cv2.dilate(img_contour, kernel, iterations=1)
    img_contour = np.flip(cv2.dilate(np.flip(img_contour), kernel, iterations=1))


    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'result_contour.png'), img_contour)
    img_contour = Image.fromarray(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    return img_contour


def MaskingMouth(img, mask, if_visual = True):
    # 生成与MaskingTeeth相同掩膜的真实嘴巴图片
    mask_mouth = np.where((mask[:,:,0]==255) & (mask[:,:,1]==255) & (mask[:,:,2]==255), 1., 0.)
    mask_mouth = np.expand_dims(mask_mouth, -1)

    img_mouth = mask_mouth * img

    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'masking_mouth.png'), img_mouth)

    return img_mouth