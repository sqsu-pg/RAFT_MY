from pyexpat import model
import sys 
sys.path.append('/home/liubo/optional_flow/RAFT/core')

import os 
import argparse
import cv2 as cv
import glob
import numpy as np 

import torch 
torch.cuda.empty_cache()
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import time


# DEVICE = 'cpu'


class RAFT_FLO:
    def __init__(self):
        self.DEVICE = 'cuda'

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', default='/home/liubo/optional_flow/RAFT/models/raft-kitti.pth', help="restore checkpoint")
        self.parser.add_argument('--path', help="dataset for evaluation")
        self.parser.add_argument('--vis_flow', action='store_true', help='use small model')#不输入--vis_flow 时候,为false, action是设置标志to true
        self.parser.add_argument('--small', action='store_true', help='use small model')#不输入--small 时候,为false, action是设置标志to true
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

        self.args = self.parser.parse_args()

        self.model = torch.nn.DataParallel(RAFT(self.args))
        # print (model)
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.to(self.DEVICE)
        self.model.eval()


    def GetFloAndReturnMat(self, Image_Pre, Image_Cur):
        with torch.no_grad():
                
            image_1 = Image_Pre
            image_2 = Image_Cur
            if len(image_1.shape) == 2:
                # print ("读入的是单通道")
                im1 = np.zeros((image_1.shape[0], image_1[1], 3))
                im2 = np.zeros((image_2.shape[0], image_2[1], 3))
                im1[:, :, 0] = image_1
                im1[:, :, 1] = image_1
                im1[:, :, 2] = image_1
                im2[:, :, 0] = image_2
                im2[:, :, 1] = image_2
                im2[:, :, 2] = image_2
                image_1 = im1
                image_2 = im2
            # start = time.time()
            image_1_tensor = torch.from_numpy(image_1).permute(2, 0, 1).float()
            image_2_tensor = torch.from_numpy(image_2).permute(2, 0, 1).float()
            # print ('image_1_tensor shape is ', image_1_tensor.shape)

            image_1_tensor = image_1_tensor[None].to(self.DEVICE)
            image_2_tensor = image_2_tensor[None].to(self.DEVICE)
            # print ('image_1_tensor[None] shape is ', image_1_tensor.shape)

            padder = InputPadder(image_1_tensor.shape)
            image1, image2 = padder.pad(image_1_tensor, image_2_tensor)

            # start1 = time.time()
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            # end1 = time.time()
            # print ("each time : ", end1 - start1)
                # print(flow_up.type(), " ", flow_up.dtype, " ", flow_up.shape)

                # dict1 = imfile1.split('/')
                # dict2 = dict1[-1].split('.')

                # save_path = ""
                # for i in range(len(dict1) - 2):
                #     if i != 0:
                #         save_path += '/'
                #     save_path += dict1[i]
                
                # save_path = save_path + "/flow/" +  dict2[0] + ".flo"
                
            flow_up = padder.unpad(flow_up)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
                # print (flo.dtype)

                # if flo.shape[0] != image_1.shape[0] or flo.shape[1] != image_1.shape[1] or flo.shape[2] != 2:
                #     print ("unpadding fail")
                # else:
                #     print("padding success")
                
                #     write_flo(save_path, flo)

            # if (args.vis_flow):
            #     flo_pic = flow_viz.flow_to_image(flo)
            #     cv.imshow('flo_img', flo_pic/255.0)
            #     cv.waitKey(10)

            # end = time.time()

            return flo

    def GetFloAndReturnMatWithVis(self, Image_Pre, Image_Cur):
        with torch.no_grad():
                
            image_1 = Image_Pre
            image_2 = Image_Cur
            if len(image_1.shape) == 2:
                # print ("读入的是单通道")
                im1 = np.zeros((image_1.shape[0], image_1[1], 3))
                im2 = np.zeros((image_2.shape[0], image_2[1], 3))
                im1[:, :, 0] = image_1
                im1[:, :, 1] = image_1
                im1[:, :, 2] = image_1
                im2[:, :, 0] = image_2
                im2[:, :, 1] = image_2
                im2[:, :, 2] = image_2
                image_1 = im1
                image_2 = im2
            # start = time.time()
            image_1_tensor = torch.from_numpy(image_1).permute(2, 0, 1).float()
            image_2_tensor = torch.from_numpy(image_2).permute(2, 0, 1).float()
            # print ('image_1_tensor shape is ', image_1_tensor.shape)

            image_1_tensor = image_1_tensor[None].to(self.DEVICE)
            image_2_tensor = image_2_tensor[None].to(self.DEVICE)
            # print ('image_1_tensor[None] shape is ', image_1_tensor.shape)

            padder = InputPadder(image_1_tensor.shape)
            image1, image2 = padder.pad(image_1_tensor, image_2_tensor)

            # start1 = time.time()
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            # end1 = time.time()
            # print ("each time : ", end1 - start1)
                # print(flow_up.type(), " ", flow_up.dtype, " ", flow_up.shape)

                # dict1 = imfile1.split('/')
                # dict2 = dict1[-1].split('.')

                # save_path = ""
                # for i in range(len(dict1) - 2):
                #     if i != 0:
                #         save_path += '/'
                #     save_path += dict1[i]
                
                # save_path = save_path + "/flow/" +  dict2[0] + ".flo"
                
            flow_up = padder.unpad(flow_up)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
                # print (flo.dtype)

                # if flo.shape[0] != image_1.shape[0] or flo.shape[1] != image_1.shape[1] or flo.shape[2] != 2:
                #     print ("unpadding fail")
                # else:
                #     print("padding success")
                
                #     write_flo(save_path, flo)

            # if (args.vis_flow):
            #     flo_pic = flow_viz.flow_to_image(flo)
            #     cv.imshow('flo_img', flo_pic/255.0)
            #     cv.waitKey(10)
            flo_pic = flow_viz.flow_to_image(flo)
            flo_bgr = np.zeros(flo_pic.shape,  dtype = np.uint8)
            ##rgb 顺序转换为 bgr
            flo_bgr[:, :, 0] = flo_pic[:, : , 2]
            flo_bgr[:, :, 1] = flo_pic[:, : , 1]
            flo_bgr[:, :, 2] = flo_pic[:, : , 0]
            flo_pic = flo_bgr
            print ("/255.0 之后: flo_pic dtype is : ", flo_pic.dtype)
            # end = time.time()

            return flo, flo_pic



if __name__ == '__main__':
    
    Image_per_path = "/home/liubo/data_sets/000000.png"
    Image_cur_path = "/home/liubo/data_sets/000001.png"

    Image_Per = cv.imread(Image_per_path)
    Image_Cur = cv.imread(Image_cur_path)

    net = RAFT_FLO()
    
    flo = net.GetFloAndReturnMat(Image_Per, Image_Cur)

    flo_pic = flow_viz.flow_to_image(flo)
    print (flo_pic.shape)
    cv.imshow('flo_img', flo_pic/255.0)
    cv.waitKey(0)
