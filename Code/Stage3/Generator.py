import torch
import numpy as np
import cv2
import os
import matplotlib
from torchvision import transforms as T
import json
import math
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab

class Mask2TeethGenerator():
    def __init__(self, network):
        super(Mask2TeethGenerator, self).__init__()
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

    def CIEDE2000_RGB(self, RGB_1, RGB_2):
        def CIEDE2000(Lab_1, Lab_2):
            '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
            C_25_7 = 6103515625 # 25**7
            
            L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
            L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
            C1 = math.sqrt(a1**2 + b1**2)
            C2 = math.sqrt(a2**2 + b2**2)
            C_ave = (C1 + C2) / 2
            G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))
            
            L1_, L2_ = L1, L2
            a1_, a2_ = (1 + G) * a1, (1 + G) * a2
            b1_, b2_ = b1, b2
            
            C1_ = math.sqrt(a1_**2 + b1_**2)
            C2_ = math.sqrt(a2_**2 + b2_**2)
            
            if b1_ == 0 and a1_ == 0: h1_ = 0
            elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
            else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi
            
            if b2_ == 0 and a2_ == 0: h2_ = 0
            elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
            else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

            dL_ = L2_ - L1_
            dC_ = C2_ - C1_    
            dh_ = h2_ - h1_
            if C1_ * C2_ == 0: dh_ = 0
            elif dh_ > math.pi: dh_ -= 2 * math.pi
            elif dh_ < -math.pi: dh_ += 2 * math.pi        
            dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)
            
            L_ave = (L1_ + L2_) / 2
            C_ave = (C1_ + C2_) / 2
            
            _dh = abs(h1_ - h2_)
            _sh = h1_ + h2_
            C1C2 = C1_ * C2_
            
            if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
            elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
            elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi 
            else: h_ave = h1_ + h2_
            
            T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)
            
            h_ave_deg = h_ave * 180 / math.pi
            if h_ave_deg < 0: h_ave_deg += 360
            elif h_ave_deg > 360: h_ave_deg -= 360
            dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))
            
            R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
            S_C = 1 + 0.045 * C_ave
            S_H = 1 + 0.015 * C_ave * T
            
            Lm50s = (L_ave - 50)**2
            S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
            R_T = -math.sin(dTheta * math.pi / 90) * R_C

            k_L, k_C, k_H = 1, 1, 1
            
            f_L = dL_ / k_L / S_L
            f_C = dC_ / k_C / S_C
            f_H = dH_ / k_H / S_H
            
            dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
            return dE_00

        rgb_1 = tuple((x /255. for x in RGB_1))
        rgb_2 = tuple((x /255. for x in RGB_2))
        Lab_1 = rgb2lab(rgb_1)
        Lab_2 = rgb2lab(rgb_2)
        dis = CIEDE2000(Lab_1, Lab_2)

        return dis

    def Mask2TeethData_Process(self, teeth_ori, teeth, mouth, mask):
        teeth_ori = cv2.cvtColor(teeth_ori, cv2.COLOR_BGR2RGB)
        teeth = cv2.cvtColor(teeth, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255

        # construct the conditional image(1)
        teeth_color_ = np.zeros(teeth.shape, dtype=np.uint8)
        teeth_id = np.zeros((teeth.shape[0], teeth.shape[1]), dtype=np.float32)
        for id in self.id_color_dict.keys():
            color = matplotlib.colors.to_rgb(self.id_color_dict[id])
            color_RGB = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            where_ori = np.where((teeth_ori[:,:,0]==color_RGB[0]) & (teeth_ori[:,:,1]==color_RGB[1]) & (teeth_ori[:,:,2]==color_RGB[2]))
            if len(where_ori[0]) == 0: # not exist this color
                continue
            where_align = np.where(np.apply_along_axis(self.CIEDE2000_RGB, 2, teeth, color_RGB) < 3)
            if len(where_align[0]) == 0: # not exist this color after alignment
                continue

            R = int(np.average(mouth[where_ori[0], where_ori[1], 0]))
            G = int(np.average(mouth[where_ori[0], where_ori[1], 1]))
            B = int(np.average(mouth[where_ori[0], where_ori[1], 2]))
            teeth_color_[where_align[0], where_align[1]] = (R, G, B)   # -> Three-channel Average Color
            teeth_id[where_align[0], where_align[1]] = int(id) / 32   # -> One-channel Teeth ID

        # img = Image.fromarray(np.uint8(teeth_color_))
        # img.save(os.path.join('./result_vis', 'teeth_average_color.png'))

        teeth = self.transform(teeth)
        mouth = self.transform(mouth)
        teeth_color = self.transform(teeth_color_)
        teeth_id = torch.from_numpy(teeth_id).unsqueeze(0)
        teeth_id = (teeth_id-0.5)*2

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth + mask*torch.randn_like(teeth)
        cond_image = torch.cat([teeth_id, teeth_color, noisy_image], dim=0)

        mask_image = teeth*(1. - mask) + mask

        out = {
            'bf_image': teeth,          #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel

            'cond_teeth_color': cv2.cvtColor(teeth_color_, cv2.COLOR_RGB2BGR),    
        }
        return out

    def Mask2TeethData_Process_ori(self, teeth_ori, teeth, mouth, mask):
        teeth_ori = cv2.cvtColor(teeth_ori, cv2.COLOR_BGR2RGB)
        teeth = cv2.cvtColor(teeth, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255

        # construct the conditional image(1)
        teeth_color_ = np.zeros(teeth.shape, dtype=np.uint8)
        teeth_id = np.zeros((teeth.shape[0], teeth.shape[1]), dtype=np.float32)
        for id in self.id_color_dict.keys():
            color = matplotlib.colors.to_rgb(self.id_color_dict[id])
            where = np.where((teeth[:,:,0]==int(color[0]*255)) & (teeth[:,:,1]==int(color[1]*255)) & (teeth[:,:,2]==int(color[2]*255)))
            if len(where[0]) == 0: # not exist this color
                continue

            R = int(np.average(mouth[where[0], where[1], 0]))
            G = int(np.average(mouth[where[0], where[1], 1]))
            B = int(np.average(mouth[where[0], where[1], 2]))
            teeth_color_[where[0], where[1]] = (R, G, B)   # -> Three-channel Average Color
            teeth_id[where[0], where[1]] = int(id) / 32   # -> One-channel Teeth ID

        # img = Image.fromarray(np.uint8(teeth_color_))
        # img.save(os.path.join('./result_vis', 'teeth_average_color.png'))

        teeth = self.transform(teeth)
        mouth = self.transform(mouth)
        teeth_color = self.transform(teeth_color_)
        teeth_id = torch.from_numpy(teeth_id).unsqueeze(0)
        teeth_id = (teeth_id-0.5)*2

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth + mask*torch.randn_like(teeth)
        cond_image = torch.cat([teeth_id, teeth_color, noisy_image], dim=0)

        mask_image = teeth*(1. - mask) + mask

        out = {
            'bf_image': teeth,          #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel  

            'cond_teeth_color': cv2.cvtColor(teeth_color_, cv2.COLOR_RGB2BGR),   
        }
        return out


    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_ori = data['crop_teeth']
            teeth = data['crop_teeth_align']
            mouth = data['crop_mouth']
            mask = data['crop_mask']

            out = self.Mask2TeethData_Process(teeth_ori, teeth, mouth, mask)
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
            return prediction, out['cond_teeth_color']


class Contour2TeethGenerator():
    def __init__(self, network):
        super(Contour2TeethGenerator, self).__init__()
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

    def Mask2TeethData_Process(self, teeth_contour, mouth, mask):
        teeth_contour = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255

        teeth_color = np.zeros(mouth.shape, dtype=np.uint8)

        # construct the conditional image(1)
        mouth_coords = np.where(mask==1)
        numof_mouth_coords = mouth_coords[0].size
        y_mouth_coords = np.random.choice(mouth_coords[0], int(numof_mouth_coords/10), replace=True)
        x_mouth_coords = np.random.choice(mouth_coords[1], int(numof_mouth_coords/10), replace=True)
        where = tuple([np.int64(y_mouth_coords), np.int64(x_mouth_coords)])
        teeth_color[where[0], where[1], :] = mouth[where[0], where[1], :]  

        # img = Image.fromarray(np.uint8(teeth_color_))
        # img.save(os.path.join('./result_vis', 'teeth_average_color.png'))

        teeth_contour = self.transform(teeth_contour)
        mouth = self.transform(mouth)
        teeth_color_ = self.transform(teeth_color)

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth_contour + mask*torch.randn_like(teeth_contour)
        cond_image = torch.cat([teeth_contour, teeth_color_, noisy_image], dim=0)

        mask_image = teeth_contour*(1. - mask) + mask

        out = {
            'bf_image': teeth_contour,  #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel
            'cond_teeth_color': cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR),    
        }
        return out

    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_contour_align = data['crop_teeth_align']
            mouth = data['crop_mouth']
            mask = data['crop_mask']

            out = self.Mask2TeethData_Process(teeth_contour_align, mouth, mask)
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
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())     # torch_BGR_uint8
            return prediction, out['cond_teeth_color']


class Contour2ToothGenerator_FaceColor_TeethColor():
    def __init__(self, network):
        super().__init__()
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

    def Mask2TeethData_Process(self, teeth_contour, mouth, mask, face):
        teeth_contour = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_color = np.uint8(face * (1. - np.expand_dims(mask, -1)))
        teeth_color = np.zeros(mouth.shape, dtype=np.uint8)

        # construct the conditional image(1)
        mouth_coords = np.where(mask==1)
        numof_mouth_coords = mouth_coords[0].size
        y_mouth_coords = np.random.choice(mouth_coords[0], int(numof_mouth_coords/10), replace=True)
        x_mouth_coords = np.random.choice(mouth_coords[1], int(numof_mouth_coords/10), replace=True)
        where = tuple([np.int64(y_mouth_coords), np.int64(x_mouth_coords)])
        teeth_color[where[0], where[1], :] = mouth[where[0], where[1], :]  

        # img = Image.fromarray(np.uint8(teeth_color_))
        # img.save(os.path.join('./result_vis', 'teeth_average_color.png'))

        teeth_contour = self.transform(teeth_contour)
        mouth = self.transform(mouth)
        teeth_color_bar = self.transform(teeth_color)
        face_color_bar = self.transform(face_color)

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth_contour + mask*torch.randn_like(teeth_contour)
        cond_image = torch.cat([teeth_contour, teeth_color_bar, face_color_bar, noisy_image], dim=0)

        mask_image = teeth_contour*(1. - mask) + mask
        
        # cv2.imwrite(os.path.join('./result_vis', 'cond_face_color.png'), cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_teeth_color.png'), cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_color.png'), cv2.cvtColor(teeth_color+face_color, cv2.COLOR_RGB2BGR))

        out = {
            'bf_image': teeth_contour,  #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel
            # 'cond_face_color': cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR),
            # 'cond_teeth_color': cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR),
            'cond_teeth_color': cv2.cvtColor(teeth_color+face_color, cv2.COLOR_RGB2BGR),
        }
        return out

    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_contour_align = data['crop_teeth_align']
            mouth = data['crop_mouth']
            mask = data['crop_mask']
            face = data['crop_face']

            out = self.Mask2TeethData_Process(teeth_contour_align, mouth, mask, face)
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
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())     # torch_BGR_uint8
            return prediction, out['cond_teeth_color']


class Contour2ToothGenerator_FaceColor_LightColor():
    def __init__(self, network):
        super().__init__()
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

    def Mask2TeethData_Process(self, teeth_contour, mouth, mask, face):
        teeth_contour = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
        mouth_gray = clahe.apply(mouth_gray)
        _, light_mask = cv2.threshold(mouth_gray, 240, 255, cv2.THRESH_BINARY)
        light_color = np.uint8(mouth * np.expand_dims(light_mask/255, -1))

        face_color = np.uint8(face * (1. - np.expand_dims(mask, -1)))
        face_light_color = light_color + face_color

        teeth_contour = self.transform(teeth_contour)
        mouth = self.transform(mouth)
        face_light_color_bar = self.transform(face_light_color)

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth_contour + mask*torch.randn_like(teeth_contour)
        cond_image = torch.cat([teeth_contour, face_light_color_bar, noisy_image], dim=0)

        mask_image = teeth_contour*(1. - mask) + mask
        
        # cv2.imwrite(os.path.join('./result_vis', 'cond_face_color.png'), cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_teeth_color.png'), cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_color.png'), cv2.cvtColor(teeth_color+face_color, cv2.COLOR_RGB2BGR))

        out = {
            'bf_image': teeth_contour,  #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel
            # 'cond_face_color': cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR),
            # 'cond_teeth_color': cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR),
            'cond_teeth_color': cv2.cvtColor(face_light_color, cv2.COLOR_RGB2BGR),
        }
        return out

    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_contour_align = data['crop_teeth_align']
            # teeth_contour_align = data['crop_teeth']
            mouth = data['crop_mouth']
            mask = data['crop_mask']
            face = data['crop_face']

            out = self.Mask2TeethData_Process(teeth_contour_align, mouth, mask, face)
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
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())     # torch_BGR_uint8
            return prediction, out['cond_teeth_color']


class Contour2ToothGenerator_Fourier():
    def __init__(self, network):
        super().__init__()
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

    def Fourier(self, img, beta):
        # calculate Fourier Parameter
        h, w, c = img.shape
        
        b = (  np.floor(np.amin((h,w))*beta)  ).astype(int)
        b_h = b
        b_w = b
        # b_h = np.floor(h*beta).astype(int)
        # b_w = np.floor(w*beta).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        h1 = c_h-b_h
        h2 = c_h+b_h+1
        w1 = c_w-b_w
        w2 = c_w+b_w+1
        # Fourier Transform
        img_fft = np.fft.fft2(img, axes=(0,1))
        amp = np.abs(img_fft)
        pha = np.angle(img_fft)
        amp_shift = np.fft.fftshift(amp, axes=(0,1))
        amp_shift_new = np.zeros((h,w,c))
        amp_shift_new[h1:h2, w1:w2, :] = amp_shift[h1:h2, w1:w2, :]
        amp_new = np.fft.ifftshift(amp_shift_new, axes=(0,1))
        recover = amp_new * np.exp( 1j * pha)
        recover = np.abs(np.fft.ifft2(recover, axes=(0,1))).astype('uint8')
        return recover, amp_shift_new

    def Mask2TeethData_Process(self, teeth_contour, mouth, mask, face):
        teeth_contour = cv2.cvtColor(teeth_contour, cv2.COLOR_BGR2RGB)
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mask = np.array(mask)[:,:,0] / 255
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        fourier_color, amp_shift_new = self.Fourier(face, beta=0.12)

        teeth_contour = self.transform(teeth_contour)
        mouth = self.transform(mouth)
        fourier_color_ = self.transform(fourier_color)

        # construct the conditional image(2)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        noisy_image = (1. - mask)*teeth_contour + mask*torch.randn_like(teeth_contour)
        cond_image = torch.cat([teeth_contour, fourier_color_, noisy_image], dim=0)

        mask_image = teeth_contour*(1. - mask) + mask
        
        # cv2.imwrite(os.path.join('./result_vis', 'cond_face_color.png'), cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_teeth_color.png'), cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join('./result_vis', 'cond_color.png'), cv2.cvtColor(teeth_color+face_color, cv2.COLOR_RGB2BGR))

        out = {
            'bf_image': teeth_contour,  #three-channel 
            'cond_image': cond_image,   #seven-channel 
            'mask': mask,               #one-channel
            'mask_image': mask_image,   #three-channel
            # 'cond_face_color': cv2.cvtColor(face_color, cv2.COLOR_RGB2BGR),
            # 'cond_teeth_color': cv2.cvtColor(teeth_color, cv2.COLOR_RGB2BGR),
            'cond_teeth_color': cv2.cvtColor(fourier_color, cv2.COLOR_RGB2BGR),
        }
        return out

    def predict(self, data):
        self.netG.eval()

        with torch.no_grad():
            teeth_contour_align = data['crop_teeth_align']
            mouth = data['crop_mouth']
            mask = data['crop_mask']
            face = data['crop_face']

            out = self.Mask2TeethData_Process(teeth_contour_align, mouth, mask, face)
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
            prediction = torch.from_numpy(self.visuals[-1].detach().float().cpu().numpy()[::-1, ...].copy())     # torch_BGR_uint8
            return prediction, out['cond_teeth_color']
