import yaml
import argparse
import os
from tqdm import tqdm
import cv2
import sys
from glob import glob
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='../Data', help='path of the input facial photograph')

    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']
    parser.set_defaults(**GeneratorConfig)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.out_path, 'processing')):
            os.makedirs(os.path.join(args.out_path, 'processing'))
    if not os.path.isdir(os.path.join(args.out_path, 'prediction')):
            os.makedirs(os.path.join(args.out_path, 'prediction'))
    if args.logger is True:
        SavedStdout = sys.stdout
        sys.stdout = open(os.path.join(args.out_path, 'error.txt'), "w")

    img_paths = natsorted(glob(os.path.join(args.img_path, '*')))

    for _, img_path in enumerate(tqdm(img_paths)):
        try:
            img_name = os.path.basename(img_path).split('.')[0]

            stage1_data = Stage1(img_path, mode=args.mode, state=args.stage1, if_visual=False)
            
            stage2_data = Stage2_Mask2Mask(stage1_data, mode=args.mode, state=args.stage2, if_visual=False)
            stage2_data.update(stage1_data)

            stage3_data = Stage3_Mask2Teeth(stage2_data, mode=args.mode, state=args.stage3, if_visual=False)
            stage3_data.update(stage2_data)


            ### save the visual results    
            for i, key in enumerate(['crop_face', 'crop_mouth', 'crop_teeth', 'crop_teeth_align', 'cond_teeth_color', 'crop_mouth_align']):
                img = stage3_data[key]          
                ### save together
                plt.subplot(3,2,i+1)
                plt.imshow(img[:,:,::-1])
                plt.axis('off')
            plt.savefig(os.path.join(os.path.join(args.out_path, 'processing'), img_name+'.png'), bbox_inches='tight')


            ### restore to original size
            pred = Restore(stage3_data['crop_mouth_align'], stage3_data)
            pred_face = pred['pred_ori_face']
            cv2.imwrite(os.path.join(os.path.join(args.out_path, 'prediction'), img_name+'.png'), pred_face)
        
        except:
            print(img_name)
            continue
        
        sys.stdout.flush()