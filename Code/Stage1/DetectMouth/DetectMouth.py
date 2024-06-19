import cv2
import os
import dlib
import numpy as np
from skimage.filters import gaussian
from .test import evaluate
import argparse
from copy import deepcopy
from PIL import Image



def mask(image, parsing, part=11, color=[139, 0, 139]):
	b, g, r = color      
	tar_color = np.zeros_like(image)
	tar_color[:, :, 0] = b
	tar_color[:, :, 1] = g
	tar_color[:, :, 2] = r

	changed = tar_color.copy()
	changed[parsing != part] = image[parsing != part]

	return changed


def DetectMouth(image):
	cp = 'Stage1/DetectMouth/cp/79999_iter.pth'
	# image = cv2.imread(image_path)

	origin_img = deepcopy(image)
	mouth_color = deepcopy(image)
	mouth_mask = np.zeros_like(image)

	parsing = evaluate(image, cp)
	parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

	part = 11  # mouth
	color = [255, 255, 255]
	mouth_color = mask(mouth_color, parsing, part, color)
	mouth_mask = mask(mouth_mask, parsing, part, color)

	# print(mouth_color.shape)
	# cv2.imshow('image', cv2.resize(origin_img, (512, 512)))
	# cv2.imshow('mask', cv2.resize(mouth_mask, (512, 512)))
	# cv2.imshow('color', cv2.resize(mouth_color, (512, 512)))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return origin_img, mouth_mask, mouth_color




def CropMouth(face, mouth_mask, crop_size=(256, 256), if_visual=True):
	
	location = np.where(mouth_mask == 255) 
	center_u = int((np.min(location[1]) + np.max(location[1])) / 2)   # u-axis is along to width
	center_v = int((np.min(location[0]) + np.max(location[0])) / 2)   # v-axis is along to height

	face = Image.fromarray(face)
	mouth_mask = Image.fromarray(mouth_mask)
	crop_face = face.crop((center_u-crop_size[0]//2, center_v-crop_size[1]//2, center_u+crop_size[0]//2, center_v+crop_size[1]//2))
	crop_mask = mouth_mask.crop((center_u-crop_size[0]//2, center_v-crop_size[1]//2, center_u+crop_size[0]//2, center_v+crop_size[1]//2))
	crop_face = np.array(crop_face)
	crop_mask = np.array(crop_mask)

	if if_visual == True:
		cv2.imwrite(os.path.join('./result_vis', 'crop_mask.png'), crop_mask)
		cv2.imwrite(os.path.join('./result_vis', 'crop_face.png'), crop_face)
	
	info = {
        'coord_x': (center_u-crop_size[0]//2, center_u+crop_size[0]//2),
        'coord_y': (center_v-crop_size[1]//2, center_v+crop_size[1]//2),
        'new_size': crop_size,
    }

	return crop_face, crop_mask, info


if __name__ == '__main__':
	img_path = './DetectMouth/images/img10.jpg'
	img = cv2.imread(img_path)
	DetectMouth(img)

