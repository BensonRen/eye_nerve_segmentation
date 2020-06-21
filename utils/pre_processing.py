"""
Pre-processing of the images

The provided images from Duke clinic is in the form of 1536 x 496
This script split and pad those images into (3 x 512) x 512 images with vertical cuts and mirror paddings
"""

import pandas as pd
import numpy as np
import os
from skimage import io
from skimage.util import pad

image_dir = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10'
output_dir = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10'
label_file_name = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10/label_file.csv'
x_len = 512

# Change all the images in image_dir into images__[0,1,2] in output_dir
for images in os.listdir(image_dir):
    # make sure this is a image
    if not images.endswith('.jpg'):
        continue
    # read the image
    img = io.imread(os.path.join(image_dir, images))
    # Split it into  3
    img_list = [img[:,:x_len], img[:, x_len:2*x_len], img[:,  2*x_len:]]
    # for each of them
    for ind, img_new in enumerate(img_list):
        # Get save name
        save_name = os.path.join(output_dir,images[:-4] + '__{}'.format(ind) + images[-4:])
        print(save_name)
        # Pad them into square
        img_paded = pad(img_new, ((8, 8), (0, 0)), mode='symmetric')
        assert np.shape(img_paded) == (x_len, x_len)
        # Save it
        io.imsave(save_name, img_paded)

# Change the names in the label file
label_file = pd.read_csv(label_file_name, header=None, index_col=0, sep=',', dtype='str', names=['labels'])
label_file.info()
labels_list = []
for i in range(3):
    labels_new = label_file.copy()
    for ind, lab in enumerate(labels_new['labels']):
        lab = lab[:-4] + '__{}'.format(i) + '.jpg'
        labels_new['labels'][ind] = lab
        #print(lab)
    #print(labels_new)
    labels_list.append(labels_new)
label_file = pd.concat(labels_list, ignore_index=True)
label_file.to_csv(label_file_name)
#print(label_file)


