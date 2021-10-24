

def count_parameters(model):
    """count trainable parameters of a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_large_nums(num):
    """add dots in large numbers to make them more readable"""
    return "{:,}".format(num).replace(",", ".")

def get_resolution(step):
    """return resolution of current step"""
    return 4 * (2**step)

def toggle_grad(model, mode):
    """turn gradient of model on/off"""
    for p in model.parameters():
        p.requires_grad = mode
