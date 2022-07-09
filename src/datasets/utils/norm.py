import torch


def global_contrast_normalisation(x: torch.FloatTensor, lambda_value=0.01, epsilon=1e-6) -> torch.FloatTensor:
    #print("x", x.shape, flush=True)

    if x.max() > 1 or x.min() < 0:
        raise ValueError("ensure image in in the range [0, 1]")

    if len(x.shape) == 5:
        batch_size, domains, channels, width, height = x.shape
        dims = (3, 4)
    elif len(x.shape) == 4:
        batch_size, channels, width, height = x.shape
        dims = (2, 3)
    elif len(x.shape) == 3:
        channels, width, height = x.shape
        dims = (1, 2)
    else:
        raise ValueError(f"unexpected image shape. got {x.shape}")

    if channels not in [3,4]:
        raise ValueError(f"unexpected image shape. got {x.shape}")

    # convert to float 
    x = x.float()

    # zero centre
    mean = x.mean(dim=dims, keepdim=True)
    #print("mean", mean.shape, flush=True)
    x = x - mean

    # normalise contrast
    contrast = (lambda_value + x.pow(2)).sqrt()
    x = x / contrast.clamp(min=epsilon)

    return x