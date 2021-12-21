import math

class Args(object):

    SEED = 42
    BATCH_SIZE = 32
    NUM_PROGRESS_IMGS = 64
    BETAS = [0.0, 0.99]
    MODEL_ID = 1 # set to none when making new checkpoint
    RESET_MODEL = False
    NOISE_DIM = 256
    FADE_SIZE = 50_000
    PHASE_SIZE = FADE_SIZE * 3
    LR = [0.0004, 0.0002, 0.0001, 0.0001, 0.0001]
    DATA_PATH = "data"
    DEVICE = "cuda"
    RESOLUTION = 64
    MAX_STEPS = int(math.log(RESOLUTION, 2))