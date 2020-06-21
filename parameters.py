"""
Params for Back propagation model
"""
# Define which data set you are using
IMG_L = 1536
IMG_W = 448
NUM_WORKERS = 5
#ROOT_DIR = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10'
#LABEL_FILE = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10/label_file.csv'
ROOT_DIR = '/work/sr365/OCT_bscans_raw/small_set10'
LABEL_FILE = '/work/sr365/OCT_bscans_raw/small_set10/label_file.csv'
#ROOT_DIR = '/work/sr365/OCT_bscans_raw/raw_bscans'
#LABEL_FILE = '/work/sr365/OCT_bscans_raw/label_file.csv'
CUT_SQUARE = True


# Training related parameters
TRAIN_STEP = 20
LEARN_RATE = 0.001
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 0.0001
BATCH_SIZE = 8
OPTIM = 'Adam'
REG_SCALE = 5e-4
EVAL_STEP = 20
MODEL_NAME = None


# Test ratio
TEST_RATIO = 0.2

