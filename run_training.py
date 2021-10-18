from train import train_model
from args import Args


def run_training():
    """
    Run training
    """
    train_model(Args)


if __name__ == "__main__":
    run_training()
    