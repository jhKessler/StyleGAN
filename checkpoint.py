import pickle

class Checkpoint(object):
    """
    class for stroring training progress
    """

    @staticmethod
    def get_filepath(Args) -> str:
        """
        Creates filepath from Args
        """
        return f"checkpoints/model_{Args.MODEL_ID}.pkl"


    @staticmethod
    def create(model_dict: dict):
        """
        Creates checkpoint object
        """
        if not Checkpoint.valid_dict(model_dict):
            raise AttributeError("Model Dict is missing one or more values")
        else:
            return Checkpoint(model_dict)


    @staticmethod
    def load(Args):
        """
        Loads model dict from file
        """
        filepath = Checkpoint.get_filepath(Args)
        # load dict
        with open(filepath, "rb") as f:
            model_dict = pickle.load(f)
        # create checkpoint object
        return Checkpoint.create(model_dict)


    @staticmethod
    def valid_dict(model_dict: dict) -> bool:
        """
        Checks if input dict has all necessary keys
        """
        dict_keys = [
            "Args",
            "generator",
            "g_optimizer",
            "discriminator",
            "d_optimizer",
            "step",
            "iteration",
            "samples",
            "start_time",
            "preview_noise",
            "alpha"
        ]
        for key in dict_keys:
            if key not in model_dict:
                return False
        return True


    def __init__(self, model_dict: dict):
        self.model_dict = model_dict


    def get_dict(self) -> dict:
        """
        Returns model dict
        """
        return self.model_dict


    def save(self, new_dict = None) -> None:
        """
        Saves model dict to file
        """
        if new_dict is not None and Checkpoint.valid_dict(new_dict):
            self.model_dict = new_dict

        filepath = Checkpoint.get_filepath(self.model_dict["Args"])

        with open(filepath, "wb") as f:
            pickle.dump(self.model_dict, f)
