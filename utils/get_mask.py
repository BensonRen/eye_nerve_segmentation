"""
This function serves to get the masked image from the label file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc
from PIL import Image
# Read part of the label file to avoid the computer RAM explosion
labels = pd.read_csv('/work/sr365/OCT_bscans_raw/20200219-segmentation_lines.csv')

img_l, img_w = 1536, 496
# Get the ilm and epr from data frame
ilm = labels[['ilm{}'.format(i+1) for i in range(img_l)]]
epr = labels[['epr{}'.format(i+1) for i in range(img_l)]]

# helper_axis
x_axis = np.arange(img_l)
y_axis = np.arange(img_w)

print("number of examples", len(labels))
#binary_mask = np.zeros([len(labels), img_l * img_w])
# Read the images according to the image file name from label file
for ind, figure_name in enumerate(labels['filejpg']):
    print("Reading picture", figure_name)
    bm = np.zeros([img_w, img_l])
    for i in range(img_l):
        mask = (y_axis+1 > ilm.iloc[ind, i]) * (y_axis + 1 < epr.iloc[ind, i])
        bm[mask, i] = 1
    # Save the mask
    np.savetxt(os.path.join('/work/sr365/OCT_bscans_raw','mask',figure_name.replace('.jpg','.csv')), bm)
