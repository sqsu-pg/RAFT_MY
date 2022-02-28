'''
Author: your name
Date: 2022-01-19 09:17:53
LastEditTime: 2022-01-21 20:34:18
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /RAFT/demo.py
'''
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
torch.cuda.empty_cache()
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import time 

torch.cuda.set_device(1)
#DEVICE = 'cpu'
DEVICE = 'cuda:1' #显卡内存不够

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()##网络输出的结果的维度应该是2 * h * w转换为h*w*2
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo fomat
    :param flow: optical flow map
    :parma filename: optical flow file path to be saved
    :return : None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

def read_flo_and_imshow(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
        print ("读入的光流的大小为: ", h, w)
    f.close()
    print ("vis")
    flo = flow_viz.flow_to_image(data2d)
    cv2.imshow('image', flo/255.0)
    cv2.waitKey()
    return data2d
        


def demo_set(args):
    model = torch.nn.DataParallel(RAFT(args))
    print (model)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    print ("current gpu : ", torch.cuda.current_device())

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            end = time.time()
            print("each time is : ", end - start)
            viz(image1, flow_up)


def make_flow_for_kitti(args):
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
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print(imfile1)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(flow_up.type(), " ", flow_up.dtype, " ", flow_up.shape)

            dict1 = imfile1.split('/')
            dict2 = dict1[-1].split('.')
            save_path = "/home/george/SLAM/VDO_SLAM/VDO-SLAM/demo-kitti/flow_raft/" + dict2[0] + ".flo"

            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            print (flo.dtype)
            write_flo(save_path, flo)
            # flow_up = torch.from_numpy(flow_up)
            # viz(image1, flow_up)

def vis_for_kitti(path):
    images = glob.glob(os.path.join(path, '*.flo'))
    images = sorted(images)

    for imfile1 in images:
        image_flo = read_flo_and_imshow(imfile1)

def make_flow_for_one_gray_kitti_picture(args, path_1, path2):

    model = torch.nn.DataParallel(RAFT(args))
    print (model)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image_1 = cv2.imread(path_1)
        image_2 = cv2.imread(path2)
        if len(image_1.shape) == 2:
            print ("读入的是单通道")
            im1 = np.zeros((image_1.shape[0], image_1[1], 3))
            im2 = np.zeros((image_2.shape[0], image_2[1], 3))
            im1[:, :, 0] = im1
            im1[:, :, 1] = im1
            im1[:, :, 2] = im1
            im2[:, :, 0] = im2
            im2[:, :, 1] = im2
            im2[:, :, 2] = im2
            image_1 = im1
            image_2 = im2
        image_1_tensor = torch.from_numpy(image_1).permute(2, 0, 1).float()
        image_2_tensor = torch.from_numpy(image_2).permute(2, 0, 1).float()
        print ('image_1_tensor shape is ', image_1_tensor.shape)

        image_1_tensor = image_1_tensor[None].to(DEVICE)
        image_2_tensor = image_2_tensor[None].to(DEVICE)
        print ('image_1_tensor[None] shape is ', image_1_tensor.shape)

        padder = InputPadder(image_1_tensor.shape)
        image1, image2 = padder.pad(image_1_tensor, image_2_tensor)
        print ('image1 shape is ', image1.shape)

        flow_low, flow_up = model(image1, image2, iters = 20, test_mode = True)

        print ("网络计算完成之后，: ")
        print("flow_up shape is : ", flow_up.shape)
        flo_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        print (flo_np.dtype)
        save_path = "/home/liubo/data_sets/03/flow/000000.flo"
        # write_flo(save_path, flo_np)
        # read_flo_and_imshow(save_path)
        save_vis_flo = "/home/liubo/data_sets/flo_vis/0000.png"
        flo = flow_viz.flow_to_image(flo_np)
        # flo = flo / 255.0
        # flo = flo.astype(np.uint8)
        # cv2.imshow('image', flo/255.0)
        cv2.imshow("image", flo)
        print ("flo dtype is ", flo.dtype)
        cv2.waitKey()
        cv2.imwrite(save_vis_flo, flo)

def make_flow_for_kitti_sets(args):
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
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image_1 = cv2.imread(imfile1)
            image_2 = cv2.imread(imfile2)
            if len(image_1.shape) == 2:
                print ("读入的是单通道")
                im1 = np.zeros((image_1.shape[0], image_1[1], 3))
                im2 = np.zeros((image_2.shape[0], image_2[1], 3))
                im1[:, :, 0] = im1
                im1[:, :, 1] = im1
                im1[:, :, 2] = im1
                im2[:, :, 0] = im2
                im2[:, :, 1] = im2
                im2[:, :, 2] = im2
                image_1 = im1
                image_2 = im2
            image_1_tensor = torch.from_numpy(image_1).permute(2, 0, 1).float()
            image_2_tensor = torch.from_numpy(image_2).permute(2, 0, 1).float()
            print ('image_1_tensor shape is ', image_1_tensor.shape)

            image_1_tensor = image_1_tensor[None].to(DEVICE)
            image_2_tensor = image_2_tensor[None].to(DEVICE)
            print ('image_1_tensor[None] shape is ', image_1_tensor.shape)

            padder = InputPadder(image_1_tensor.shape)
            image1, image2 = padder.pad(image_1_tensor, image_2_tensor)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(flow_up.type(), " ", flow_up.dtype, " ", flow_up.shape)

            dict1 = imfile1.split('/')
            dict2 = dict1[-1].split('.')

            save_path = ""
            for i in range(len(dict1) - 2):
                if i != 0:
                    save_path += '/'
                save_path += dict1[i]
            
            save_path = save_path + "/flow/" +  dict2[0] + ".flo"
            
            flow_up = padder.unpad(flow_up)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            print (flo.dtype)

            if flo.shape[0] != image_1.shape[0] or flo.shape[1] != image_1.shape[1] or flo.shape[2] != 2:
                print ("unpadding fail")
            else:
                print("padding success")
            
                write_flo(save_path, flo)
                # flow_up = torch.from_numpy(flow_up)
                # viz(image1, flow_up)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')#不输入--small 时候,为false, action是设置标志to true
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # demo_set(args)
    # make_flow_for_kitti(args)#python demo.py --model=models/raft-kitti.pth --path=
    # vis_for_kitti("/home/george/data_sets/03/flow/")

    path1 = "/home/liubo/data_sets/03/image_0/000000.png"
    path2 = "/home/liubo/data_sets/03/image_0/000001.png"

    make_flow_for_one_gray_kitti_picture(args, path1, path2)
    #make_flow_for_kitti_sets(args)
