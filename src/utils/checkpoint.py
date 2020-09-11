import os
import shutil
import torch


def save_checkpoint(state, is_best, root, filename):
    """
    Saves checkpoint and best checkpoint (optionally)
    """
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(
                root, filename), os.path.join(
                root, 'best_' + filename))