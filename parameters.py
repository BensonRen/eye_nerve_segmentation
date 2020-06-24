"""
Params for Back propagation model
"""
# Define which data set you are using
#IMG_L = 1536
#IMG_W = 448
IMG_L = 512
IMG_W = 512
NUM_WORKERS = 8
CUT_SQUARE = False
PRETRAIN = False
# Mac set
#ROOT_DIR = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10'
#LABEL_FILE = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10/label_file.csv'

# Small set
#TRAIN_ROOT_DIR = '/work/sr365/OCT_bscans_raw/small_set10/train'
#TRAIN_LABEL_FILE = '/work/sr365/OCT_bscans_raw/small_set10/train/label_file.csv'
#TEST_ROOT_DIR = '/work/sr365/OCT_bscans_raw/small_set10/test'
#TEST_LABEL_FILE = '/work/sr365/OCT_bscans_raw/small_set10/test/label_file.csv'

# Big set
TRAIN_ROOT_DIR = '/work/sr365/OCT_bscans_raw/train'
TRAIN_LABEL_FILE = '/work/sr365/OCT_bscans_raw/train/label_file.csv'
TEST_ROOT_DIR = '/work/sr365/OCT_bscans_raw/test'
TEST_LABEL_FILE = '/work/sr365/OCT_bscans_raw/test/label_file.csv'


# Training related parameters
TRAIN_STEP = 5
MAX_TRAIN_SAMPLE = 5000
MAX_TEST_SAMPLE = 40
LEARN_RATE = 0.001
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 0.0001
BATCH_SIZE = 2
OPTIM = 'Adam'
REG_SCALE = 5e-4
EVAL_STEP = 20
MODEL_NAME = None


# Test ratio (Currently this flag value is useless because the patien id issue)
TEST_RATIO = 0.2

