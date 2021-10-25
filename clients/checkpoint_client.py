import os
import torch
import torch.nn as nn
import time

from . import ImageClient
from checkpoint import Checkpoint
from models import Generator
from models import Discriminator


class CheckpointClient(object):
    """
    Checkpoint Client providing functionality for creating and interacting with Checkpoints
    """

    @staticmethod
    def id_is_valid(model_id: int, test = False) -> bool:
        """
        Checks if given model id is valid
        """

        if not isinstance(model_id, int) or model_id < 0:
            return False

        folder_path = "checkpoints" if not test else "tests"
        all_models = [path for path in os.listdir(folder_path) if path.endswith(".pkl")]
        all_models = [int(path.replace(".pkl", "").split("_")[-1]) for path in all_models]
        return model_id in all_models


    @staticmethod
    def create_model_id(test = False) -> int:
        """
        Creates valid model id for new checkpoint
        """

        folder_path = "checkpoints" if not test else "tests"
        all_models = [path for path in os.listdir(folder_path) if path.endswith(".pkl")]
        all_models = [int(path.replace(".pkl", "").split("_")[-1]) for path in all_models]

        # return 1 if no checkpoint exists
        if not all_models:
            return 1

        # loop until id is not taken
        for i in range(1, max(all_models) + 2):
            if i not in all_models:
                return i


    def get_checkpoint(Args) -> Checkpoint:
        """
        returns model dict containing all relevant checkpoint attributes
        """

        # create new model if model id is None
        if Args.MODEL_ID is None and Args.RESET_MODEL:
            raise RuntimeError("Cannot reset model if Model id is not specified")
        elif Args.MODEL_ID or Args.RESET_MODEL:
                # create default values for training
                Args.MODEL_ID = CheckpointClient.create_model_id() if Args.MODEL_ID is None else Args.MODEL_ID

                generator = Generator(style_dim = Args.NOISE_DIM).to(Args.DEVICE)
                generator.apply(CheckpointClient.weights_init)
                g_optimizer = torch.optim.Adam(generator.parameters(), Args.LR, Args.BETAS)

                discriminator = Discriminator(bias = False).to(Args.DEVICE)
                discriminator.apply(CheckpointClient.weights_init)
                d_optimizer = torch.optim.Adam(discriminator.parameters(), Args.LR, Args.BETAS)

                # set training progress data
                step = iteration = samples = 0
                start_time = time.time()

                # create preview noise for showing progress
                preview_noise = ImageClient.make_image_noise(Args.NUM_PROGRESS_IMGS, Args.NOISE_DIM, Args.DEVICE)

                model_dict = {
                    "Args": Args,
                    "generator": generator,
                    "g_optimizer": g_optimizer,
                    "discriminator": discriminator,
                    "d_optimizer": d_optimizer,
                    "step": step,
                    "iteration": iteration,
                    "samples": samples,
                    "start_time": start_time,
                    "preview_noise": preview_noise,
                    "alpha": 1
                }
                return Checkpoint.create(model_dict = model_dict)

        elif not CheckpointClient.id_is_valid(Args.MODEL_ID):
            raise ValueError("Model id does not exist, set model id to None when creating new model")
        
        else:
            return Checkpoint.load(Args)


    @staticmethod
    def weights_init(layer):
        
        if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.kaiming_normal_(layer.weight)
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)