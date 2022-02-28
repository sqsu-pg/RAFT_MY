import sys 
sys.path.append('core')

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

DEVICE = 'cuda'
# DEVICE = 'cpu'

def test_raft_time_kitti(args):
    model = torch.nn.DataParallel(RAFT(args))
    print (model)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        print(len(images))

        sum_of_raft_time = 0
        sum_of_raft_img = len(images)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            
            image_1 = cv.imread(imfile1)
            image_2 = cv.imread(imfile2)
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
            start = time.time()
            image_1_tensor = torch.from_numpy(image_1).permute(2, 0, 1).float()
            image_2_tensor = torch.from_numpy(image_2).permute(2, 0, 1).float()
            # print ('image_1_tensor shape is ', image_1_tensor.shape)

            image_1_tensor = image_1_tensor[None].to(DEVICE)
            image_2_tensor = image_2_tensor[None].to(DEVICE)
            # print ('image_1_tensor[None] shape is ', image_1_tensor.shape)

            padder = InputPadder(image_1_tensor.shape)
            image1, image2 = padder.pad(image_1_tensor, image_2_tensor)

            start1 = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            end1 = time.time()
            print ("each time : ", end1 - start1)
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

            if (args.vis_flow):
                flo_pic = flow_viz.flow_to_image(flo)
                cv.imshow('flo_img', flo_pic/255.0)
                cv.waitKey(10)

            end = time.time()

            sum_of_raft_time = sum_of_raft_time + end - start
        
        mean_raft_time = sum_of_raft_time / sum_of_raft_img
        Hz_raft = 1 / mean_raft_time

        print ("mean_raft_time is : ", mean_raft_time)
        print ("Hz_raft is : ", Hz_raft)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--vis_flow', action='store_true', help='use small model')#不输入--vis_flow 时候,为false, action是设置标志to true
    parser.add_argument('--small', action='store_true', help='use small model')#不输入--small 时候,为false, action是设置标志to true
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    test_raft_time_kitti(args)