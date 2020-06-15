"""
Params for Back propagation model
"""
# Define which data set you are using
IMG_L = 1536
IMG_W = 496


# Training related parameters
TRAIN_STEP = 500
LEARN_RATE = 0.001
LR_DECARY_RATE = 0.5
STOP_THRESHOLD = 0.0001
OPTIM = 'Adam'
REG_SCALE = 5e-4
EVAL_STEP = 20
MODEL_NAME = None


# Test ratio
TEST_RATIO = 0.2

