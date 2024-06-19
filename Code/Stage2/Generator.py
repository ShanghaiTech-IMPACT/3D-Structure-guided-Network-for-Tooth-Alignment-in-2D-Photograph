import torch
import numpy as np
import cv2
import matplotlib
from torchvision import transforms as T
import json
from torchvision import transforms

class Mask2MaskGenerator():
    def __init__(self, network):
        super(Mask2MaskGenerator, self).__init__()
        self.netG = network.cuda()
        self.netG.set_new_noise_schedule()
        
        with open('./config/id_color_dict.json', 'r') as f:
            self.id_color_dict = json.load(f)['id_color_dict']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.bf_image = data.get('bf_image').cuda()
        self.cond_image = data.get('cond_image').cuda()
        self.mask = data.get('mask').cuda()
        self.mask_image = data.get('mask_image')


    def Mask2MaskData_Process(self, teeth_mask, mouth_mask):
        '''
        teeth_mask: numpy_cv2(BGR)
        mouth_mask: numpy_cv2(BGR)
        '''
        bf_data = cv2.cvtColor(teeth_mask, cv2.COLOR_BGR2RGB)    # numpy_RGB_uint8
        mask = np.array(mouth_mask)[:,:,0] / 255

        # construct the conditional image
        teeth_id = np.zeros((bf_data.shape[0], bf_data.shape[1]), dtype=np.float32)
        for id in self.id_color_dict.keys():
            color = matplotlib.colors.to_rgb(self.id_color_dict[id])
            where = np.where((bf_data[:,:,0]==int(color[0]*255)) & (bf_data[:,:,1]==int(color[1]*255)) & (bf_data[:,:,2]==int(color[2]*255)))
            if len(where[0]) == 0: # not exist this color
                continue

            teeth_id[where[0], where[1]] = int(id) / 32   # -> One-channel Teeth ID

        bf_data = self.transform(bf_data)
        teeth_id = torch.from_numpy(teeth_id).unsqueeze(0)
        teeth_id = (teeth_id-0.5)*2

        # construct the conditional image
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*bf_data + mask*torch.randn_like(bf_data)
        cond_image = torch.cat([teeth_id, noisy_image], dim=0)

        mask_image = bf_data*(1. - mask) + mask

        out = {
            'bf_image': bf_data,        #three-channel 
            'cond_image': cond_image,   #four-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel    
        }
        return out


    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_mask = data['crop_teeth']
            mouth_mask = data['crop_mask']

            out = self.Mask2MaskData_Process(teeth_mask, mouth_mask)
            bf_image = out['bf_image']
            cond_image = out['cond_image']
            mask_image = out['mask_image']
            mask = out['mask']

            self.set_input({
                'bf_image': bf_image.unsqueeze(0),
                'cond_image': cond_image.unsqueeze(0),   #four-channel 
                'mask': mask.unsqueeze(0),               #one-channel
                'mask_image': mask_image.unsqueeze(0),   #three-channel      
            })

            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=torch.randn_like(self.bf_image), y_0=self.bf_image, mask=self.mask, sample_num=1)
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())   # torch_BGR_uint8
            return prediction


class Contour2ContourGenerator():
    def __init__(self, network):
        super(Contour2ContourGenerator, self).__init__()
        self.netG = network.cuda()
        self.netG.set_new_noise_schedule()   

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.bf_image = data.get('bf_image').cuda()
        self.cond_image = data.get('cond_image').cuda()
        self.mask = data.get('mask').cuda()
        self.mask_image = data.get('mask_image')


    def Contour2ContourData_Process(self, teeth_contour, mouth_mask):
        '''
        teeth_contour: numpy_cv2(BGR)
        mouth_mask: numpy_cv2(BGR)
        '''
        bf_data = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mask = np.array(mouth_mask)[:,:,0] / 255

        bf_data = self.transform(bf_data)

        # construct the conditional image
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        c, h, w = bf_data.shape
        noisy_image = (1. - mask)*bf_data + mask*torch.randn((3, h, w))
        cond_image = torch.cat([bf_data, noisy_image], dim=0)
        mask_image = bf_data*(1. - mask) + mask

        out = {
            'bf_image': bf_data,        #three-channel 
            'cond_image': cond_image,   #four-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel    
        }
        return out


    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_contour = data['crop_teeth']
            mouth_mask = data['crop_mask']

            out = self.Contour2ContourData_Process(teeth_contour, mouth_mask)
            bf_image = out['bf_image']
            cond_image = out['cond_image']
            mask_image = out['mask_image']
            mask = out['mask']

            self.set_input({
                'bf_image': bf_image.unsqueeze(0),
                'cond_image': cond_image.unsqueeze(0),   #four-channel 
                'mask': mask.unsqueeze(0),               #one-channel
                'mask_image': mask_image.unsqueeze(0),   #three-channel      
            })

            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=torch.randn_like(self.bf_image), y_0=self.bf_image, mask=self.mask, sample_num=1)
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())
            return prediction
