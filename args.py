import math

class Args(object):

    SEED = 42
    BATCH_SIZE = 32
    NUM_PROGRESS_IMGS = 64
    BETAS = [0.0, 0.99]
    MODEL_ID = None # set to none when making new checkpoint
    RESET_MODEL = False
    NOISE_DIM = 256
    FADE_SIZE = 1_200_000
    PHASE_SIZE = FADE_SIZE * 2
    LR = 0.0002
    DATA_PATH = "data"
    DEVICE = "cuda"
    RESOLUTION = 64
    MAX_STEPS = int(math.log(RESOLUTION, 2))