from Stage1.DetectFace.DetectFace import DetectFace
from Stage1.DetectMouth.DetectMouth import DetectMouth, CropMouth
from Stage1.SegmentTeeth.DetectContour import MaskingMouth
from Stage1.SegmentToothContour.SegmentToothContour import SegmentToothContour
import numpy as np


def Stage1(img_path, mode, state, if_visual=False):
    ###################################################################################
    # stage 1
    # face detecttion and crop mouth
    ###################################################################################
    ori_img, face, info_detectface = DetectFace(img_path, newsize=(512, 512))
    face, mouth_mask, mouth_color = DetectMouth(face)
    crop_face, crop_mask, info_cropmouth = CropMouth(face, mouth_mask, crop_size=(256, 128), if_visual=if_visual)          # crop_face.png & crop.mask.png

    if mode in ['C2C2T_v2', 'C2C2T_v2_facecolor_teethcolor', 'C2C2T_v2_facecolor_lightcolor', 'C2C2T_v2_fourier']:
        ###################################################################################
        # stage 2
        # tooth contour segmentation
        mouth_masking = MaskingMouth(crop_face, crop_mask, if_visual=if_visual)                                # masking_mouth.png
        mouth_masking = np.uint8(mouth_masking)
        ###################################################################################
        teeth_contour = SegmentToothContour(mouth_masking, state, if_visual=if_visual)
        teeth_contour = np.uint8(crop_mask/255 * teeth_contour)
        crop_teeth = teeth_contour
    

    return {
        "ori_face": ori_img,          #numpy_BGR_uint8
        "detect_face": face,          #numpy_BGR_uint8

        "info": {0: info_detectface, 1: info_cropmouth},

        "crop_face": crop_face,       #numpy_BGR_uint8
        "crop_mouth": mouth_masking,  #numpy_BGR_uint8
        "crop_teeth": crop_teeth,     #numpy_BGR_uint8
        "crop_mask": crop_mask,       #numpy_BGR_uint8
    }


if __name__ == "__main__":
    img_path = r'C:\IDEA_Lab\Project_tooth_photo\Img2Img\Data\118_199fcc33faec4b39bb0fe2efc9e09cf3.jpg'
    mode = 1
    state = "Stage1/ToothContourDetect/ckpt/ckpt_4800.pth"
