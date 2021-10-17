import os



def id_is_valid(model_id: int, test = False) -> bool:
    """
    Checks if given model id is valid
    """

    if not isinstance(model_id, int):
        return False

    folder_path = "checkpoints" if not test else "tests"
    all_models = [path for path in os.listdir(folder_path) if path.endswith(".pkl")]
    all_models = [int(path.replace(".pkl", "").split("_")[-1]) for path in all_models]
    return model_id in all_models



def create_model_id(test = False):
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