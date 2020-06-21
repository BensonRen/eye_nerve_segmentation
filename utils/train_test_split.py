"""
This is the manual effort to split the training and test set
As instructed by Dr. Mariottoni, training and test set needs to be split by patient id,
which is pretty hard to do at random splits. Therefore, a manual split would be performed here
due to uneven distribution / histogram of the patient id
The training and test split ratio is around 80/20 and the index to split is 5481
"""
import pandas as pd
import shutil
import os

# Define some names and constants
file_name =  '/work/sr365/OCT_bscans_raw/label_file.csv'
img_dir = '/work/sr365/OCT_bscans_raw/raw_bscans'
target_test_folder = '/work/sr365/OCT_bscans_raw/test'
test_index = 5480

# Read the file
labels = pd.read_csv('/work/sr365/OCT_bscans_raw/20200219-segmentation_lines.csv')
for i in range(test_index):
    shutil.move(os.path.join(img_dir, labels['filejpg'][i]),  os.path.join(target_test_folder, labels['filejpg'][i]))

# Change the label file
label_file = pd.read_csv(file_name, index_col=0)
label_file.iloc[:test_index,:].to_csv(os.path.join(target_test_folder,'label_file.csv'))

