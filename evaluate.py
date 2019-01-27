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
from deeplab.datasets import BerkeleyDataTestSet
from deeplab.datasets import CityscapesDataSet
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


# DATA_DIRECTORY = './CityScapes/stuttgart_01/'
# DATA_LIST_PATH = './dataset/list/CityScapesStuttgart_01.txt'
# DATA_DIRECTORY = './SG_Driving/'
# DATA_LIST_PATH = './dataset/list/SG_Driving.txt'
DATA_DIRECTORY = './BDD_Deepdrive/bdd100k/'
DATA_LIST_PATH = './dataset/list/BDD_test.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 3
RESTORE_FROM = './BDD_20000.pth'

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
    import multiprocessing as mp
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = mp.Pool()
    # Mapping the function f to the dataset
    '''
    https://hg.python.org/cpython/file/2.7/Lib/multiprocessing/pool.py
    States that chunksize, extra = samples // 4 * num_workers
    if extra: 
        chunksize += 1

    '''
    m_list = pool.map(f, data_list, chunksize=625) # Validation set 10,000 length
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

def overlay(name, original_directory, masked_directory):
    # Save individual images with respective mask as frames for video
    # alpha = 0.30 # for singapore and cityscapes
    alpha = 0.15
    original_image = cv2.imread(original_directory + name[0] + '.jpg')
    masked = cv2.imread(masked_directory + name[0] + '_masked.png')


    cv2.addWeighted(masked, alpha, original_image, 1 - alpha, 0, original_image)

    cv2.imshow("view", original_image)


    cv2.imwrite("./BDD_Overlayed/" + name[0] + "_overlayed.jpg", original_image)




def show_all(pred, ground_truth=None, name=None):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'cDrivable', 'altLane'))

    colormap = [(0.96,0.86,0.7), (0,0,0.5), (0,0.5,0)]


    cmap = colors.ListedColormap(colormap)
    bounds = [0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imsave("./BDDMasked/" + name[0] + "_masked", pred, cmap=cmap)
    plt.close()

    # ax1.set_title('ground_truth')
    # ax1.imshow(ground_truth, cmap=cmap, norm=norm)

    # ax2.set_title('pred')
    # ax2.imshow(pred, cmap=cmap, norm=norm)
    # plt.show(fig)

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    interp = nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)
    data_list = []



    '''
    Use the following Data Directories and List Path for validation (BDD Dataset)
    DATA_DIRECTORY = './BDD_Deepdrive/bdd100k/'
    DATA_LIST_PATH = './dataset/list/BDD_val.txt'

    '''
    # valid_loader = data.DataLoader(BerkeleyDataSet(args.data_dir, args.data_list, mean=IMG_MEAN, scale=False, mirror=False, train=False), 
                                    # batch_size=1, shuffle=False, pin_memory=True)

    #Evaluation loop for Valid Loader 
    # with torch.no_grad():
    #     for index, batch in enumerate(valid_loader):     
    #         if index % 100 == 0:
    #             print('%d processd'%(index))
    #         image, label, name, size = batch

    #         h, w, c = size[0].numpy()
    #         # print(name)

    #         output = model(Variable(image).cuda(gpu0))
    #         output = interp(output).cpu().data[0].numpy()
    #         # print(output.shape)
    #         output = output.transpose(1,2,0)


    #         output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

    #         ground_truth = np.asarray(label[0].numpy()[:h,:w], dtype=np.int)


    #         show_all(ground_truth, output, name)
    #         data_list.append([ground_truth.flatten(), output.flatten()])

        # get_iou(data_list, args.num_classes)

    '''

    Use the following codes on the testset.


    Use the following Data Directories and List Path for testing (BerkeleyDataTestSet)
    DATA_DIRECTORY = './BDD_Deepdrive/bdd100k/'
    DATA_LIST_PATH = './dataset/list/BDD_test.txt'
    test_loader = data.DataLoader(BerkeleyDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN))
    

    Use the following for Cityscapes Dataset (CityscapesDataSet)
    DATA_DIRECTORY = './CityScapes/stuttgart_01/'
    DATA_LIST_PATH = './dataset/list/CityScapesStuttgart_01.txt'
    test_loader = data.DataLoader(CityscapesDataSet(args.data_dir, args.data_list, mean=IMG_MEAN))

    
    '''


    test_loader = data.DataLoader(BerkeleyDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN))
    masked_directory = 'D:/PyTorch-DeepLab-Berkeley/BDDMasked/'
    # #Evaluation loop for Test Loader 
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            if index % 100 == 0:
                print('%d processd'%(index))
            image, name, size = batch
            h, w, c = size[0].numpy()
            # print(name)

            output = model(Variable(image).cuda(gpu0))
            output = interp(output).cpu().data[0].numpy()

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            # show_all(output, name=name)
            overlay(name=name, original_directory='./BDD_Deepdrive/bdd100k/images/100k/test/', masked_directory=masked_directory)


    

if __name__ == '__main__':
    main()
