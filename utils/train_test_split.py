"""
This is the manual effort to split the training and test set
As instructed by Dr. Mariottoni, training and test set needs to be split by patient id,
which is pretty hard to do at random splits. Therefore, a manual split would be performed here
due to uneven distribution / histogram of the patient id
The training and test split ratio is around 80/20 and the index to split is 5480
"""
import pandas as pd
import shutil
import os

# Define some names and constants
label_file = ''
img_dir = ''
target_test_folder = ''
test_index = 5480

# Read the file
labels = pd.read_csv('20200219-segmentation_lines.csv')
for i in range(5480):
    shutil.move(os.path.join(img_dir, labels['filejpg'][i]),  os.path.join(target_test_folder, labels['gilejpg'][i]))


