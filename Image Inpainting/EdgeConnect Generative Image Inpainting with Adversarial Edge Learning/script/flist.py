import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../../data/mask', help='path to the dataset')
parser.add_argument('--output', type=str, default='../../../data/edge_connect_flist/mask.flist', help='path to the file list')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

images = []
for root, dirs, files in os.walk(args.path):
    # ------------------------------------------------------------
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # ------------------------------------------------------------
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            # -----------------------------------------------
            # os.path.splitext():
            # 分离文件名与扩展名；默认返回(fname,fextension)元组
            # -----------------------------------------------
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
# -------------------
# np.savetxt()
# 将array保存到txt文件
# -------------------