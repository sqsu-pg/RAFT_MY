
import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np

from utils import flow_viz

  

def read_imshow_and_save_vis_flo_png(args, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    filename = args.flo_path
    if memcached:
        filename = io.BytesIO(args.flo_path)
    
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
    cv2.imwrite(args.save_visflo_path, flo)
    cv2.imshow('image', flo/255.0)
    cv2.waitKey()
    return data2d



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_visflo_path', help="restore checkpoint")
    parser.add_argument('--flo_path', help="dataset for evaluation")
    args = parser.parse_args()

    read_imshow_and_save_vis_flo_png(args)

