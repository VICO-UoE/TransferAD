import numpy as np
import itertools
import os
import shutil
import sys
import time
import torch

newline = lambda f: print("", file=f, flush=True)
to_gpu = lambda *tensors: map(torch.Tensor.cuda, tensors)
ul = lambda s: "\x1b[4m%s\x1b[0m" % s
ul_warn = lambda s: "\x1b[4;33m%s\x1b[0m" % s


def checkpoint(path, f, name="f"):
    """
    Store model for subsequent use.
    """
    torch.save(f.state_dict(), os.path.join(path, f"{name}.pth"))


class dotdict(dict):
    """
    dot.notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def vars(self):
        return self.__dic__


def make_dirs(*args):
    for dir in args:
        os.makedirs(dir, exist_ok=True)


def save_model_info(config, path=None, file=None):
    passed_cmds = [k.replace("--", "") for k in sys.argv[1:] if '--' in k]
    config_dict = vars(config)

    for j, (k, i) in enumerate(config_dict.items()):
        print("%s: %s" % (k, ul_warn(i) if k in passed_cmds else ul(i)),
            file=file,
            flush=True)
    newline(file)

    if path is None:
        path = config.ckpt_path

    with open(os.path.join(path, "preprocessor.dat"), "w") as f:
        f.write(str(config_dict))
