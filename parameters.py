"""
Params for Back propagation model
"""
# Define which data set you are using
IMG_L = 1536
IMG_W = 448
ROOT_DIR = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10'
LABEL_FILE = '/Users/ben/Downloads/Eye segmentation project/OCT_bscans_raw/small_set10/label_file.csv'
CUT_SQUARE = True


# Training related parameters
TRAIN_STEP = 500
LEARN_RATE = 0.001
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 0.0001
BATCH_SIZE = 5
OPTIM = 'Adam'
REG_SCALE = 5e-4
EVAL_STEP = 20
MODEL_NAME = None


# Test ratio
TEST_RATIO = 0.2

