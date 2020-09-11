import torch
from timm.optim import RMSpropTF


def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    opt_name = cfg.OPT.OPTIMIZER
    eps = cfg.OPT.EPS

    params = []
    for key, value in model.named_parameters():
        lr = cfg.OPT.BASE_LR
        if not value.requires_grad:
            continue
        weight_decay = cfg.OPT.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr, eps=eps)
    elif opt_name == "rmsprop":
        optimizer = RMSpropTF(params, lr, alpha=0.9, momentum=0.9, eps=eps)
    return optimizer