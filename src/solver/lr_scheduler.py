import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def make_lr_scheduler(cfg, optimizer, train_loader):
    num_epochs = cfg.TRAIN.EPOCHS
    num_decay_epochs = cfg.OPT.DECAY_EPOCHS
    decay_rate = cfg.OPT.DECAY_RATE
    num_warmup_epochs = cfg.OPT.WARMUP_EPOCHS
    grad_acc_steps = cfg.OPT.GD_STEPS
    schedule = cfg.OPT.SCHED
    assert schedule in ["constant", "constant_warmup", "cosine_warmup", "decay_warmup"], \
        "Unsupported learning rate scheduler."

    num_training_batches = len(train_loader)
    num_training_steps = (num_epochs * num_training_batches) // grad_acc_steps
    num_warmup_steps = (num_warmup_epochs * num_training_batches) // grad_acc_steps

    if schedule == "constant":
        lr_scheduler = get_constant_schedule(optimizer)
    elif schedule == "constant_warmup":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer,
            num_warmup_steps)
    elif schedule == "cosine_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps, num_training_steps)
    elif schedule == "decay_warmup":
        num_decay_steps = (num_decay_epochs * num_training_batches) // grad_acc_steps
        lr_scheduler = get_decay_schedule_with_warmup(optimizer, num_training_steps,
            num_warmup_steps, num_decay_steps, decay_rate)

    return lr_scheduler


def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_decay_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps, num_decay_steps, gamma=0.97, last_epoch=-1):
    """ Create a schedule with learning rate that decays by `gamma`
    every `num_decay_steps`.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        decay_factor = gamma ** math.floor(current_step / num_decay_steps)
        return decay_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)