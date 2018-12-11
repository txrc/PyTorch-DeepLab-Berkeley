import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from deeplab.model import Res_Deeplab
from deeplab.datasets import BerkeleyDataSet
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './BDD_Deepdrive/bdd100k/'
DATA_LIST_PATH = './dataset/list/BDD_val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 3
NUM_STEPS = 10000 # Number of images in the validation set.
RESTORE_FROM = './Berkeley_scenes_20000.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
     A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')

def show_all(ground_truth, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'cDrivable', 'altLane'))

    colormap = [(0,0,0), (0.5,0,0), (0,0.5,0)]

    # classes = np.array(('background',  # always index 0
    #            'aeroplane', 'bicycle', 'bird', 'boat',
    #            'bottle', 'bus', 'car', 'cat', 'chair',
    #                      'cow', 'diningtable', 'dog', 'horse',
    #                      'motorbike', 'person', 'pottedplant',
    #                      'sheep', 'sofa', 'train', 'tvmonitor'))


    # colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
    #                 (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
    #                 (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
    #                 (0.5,0.75,0),(0,0.25,0.5)]

    cmap = colors.ListedColormap(colormap)
    bounds = [0,1,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('ground_truth')
    ax1.imshow(ground_truth, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)
    plt.show(fig)

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(BerkeleyDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False, train=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)

    # testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False), 
    #                                 batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
    data_list = []
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd'%(index))
            image, label, name, size = batch

            h, w, c = size[0].numpy()
            print(name)

            output = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp(output).cpu().data[0].numpy()
            print(output.shape)
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            print(output)
            print(label[0].shape)
            ground_truth = np.asarray(label[0].numpy()[:h,:w], dtype=np.int)
            

            show_all(ground_truth, output)
            data_list.append([ground_truth.flatten(), output.flatten()])

        get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()
