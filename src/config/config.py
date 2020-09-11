from yacs.config import CfgNode


global_cfg = CfgNode()


def get_cfg():
    """
    Get a copy of the default config.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg):
    """
    Let the global config point to the given cfg.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)