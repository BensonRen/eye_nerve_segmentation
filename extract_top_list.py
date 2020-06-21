import pandas as pd

# Read in the labels
n = 20
labels = pd.read_csv('/work/sr365/OCT_bscans_raw/20200219-segmentation_lines.csv',nrows=n)

labels.to_csv('/work/sr365/OCT_bscans_raw/top_{}_labels.csv'.format(n))
