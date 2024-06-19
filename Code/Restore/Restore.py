import numpy as np
import cv2
import os

def Restore(mouth_align, data):
    ori_face = data['ori_face'].copy()
    detect_face = data['detect_face'].copy()
    crop_face = data['crop_face'].copy()
    crop_mask = data['crop_mask']
    info = data['info']

    info_0, info_1 = info[0], info[1]
    face_coord_x = info_0['coord_x']
    face_coord_y = info_0['coord_y']
    face_size = info_0['face_size']
    face_new_size = info_0['new_size']

    mouth_coord_x = info_1['coord_x']
    mouth_coord_y = info_1['coord_y']
    mouth_new_size = info_1['new_size']

    # crop_face with new teeth (numpy_BGR_uint8_256*128)
    where = np.where((crop_mask[:,:,0]==255) & (crop_mask[:,:,1]==255) & (crop_mask[:,:,2]==255))
    crop_face[where[0], where[1]] = mouth_align[where[0], where[1]]

    # detect_face with new teeth (numpy_BGR_uint8_512*512)
    detect_face[mouth_coord_y[0]:mouth_coord_y[1], mouth_coord_x[0]:mouth_coord_x[1], :] = crop_face

    # ori_face with new teeth (numpy_BGR_uint8_orisize)
    detect_face_resize = cv2.resize(detect_face, face_size)
    ori_face[face_coord_y[0]:face_coord_y[1], face_coord_x[0]:face_coord_x[1], :] = detect_face_resize


    return {
        "pred_ori_face": ori_face,          #numpy_BGR_uint8_orisize
        "pred_detect_face": detect_face,    #numpy_BGR_uint8_512*512
        "pred_crop_face": crop_face,        #numpy_BGR_uint8_256*128
    }





