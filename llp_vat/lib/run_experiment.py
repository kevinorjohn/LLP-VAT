import glob
import os
import pathlib
import warnings

import logzero
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter


def write_meters(epoch, tag, tb_writer, meters):
    for name, value in meters.averages("").items():
        tb_writer.add_scalar("{}/{}".format(tag, name), value, epoch)


def save_checkpoint(filename, model, epoch, optimizer=None):
    checkpoint = {'epoch': epoch}
    if isinstance(model, nn.DataParallel):
        checkpoint['state_dict'] = model.module.state_dict()
    else:
        checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    else:
        return model


class RunExperiment:
    def __init__(self, result_dir, mode="w"):
        self._check_path(result_dir)
        self.result_dir = result_dir
        self.mode = mode

    def _check_path(self, path):
        """Create directory if path doesn't exist"""
        if path is not None:
            if os.path.isfile(path):
                raise TypeError("Cannot create directory {}".format(path))
            target_dir = path

            if os.path.exists(path):
                warnings.warn(
                    "Experiment {} has been executed before".format(path))
                opt = input("Continue running the experiment, y/[n]: ")
                if opt.lower() != "y":
                    raise RuntimeError()
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    def create_logfile(self, name):
        fmt = ("%(color)s[%(levelname)s %(name)s %(asctime)s]"
               "%(end_color)s %(message)s")
        log_fmt = logzero.LogFormatter(fmt=fmt)

        if name is None:
            filename = None
        elif not name.endswith(".log"):
            filename = os.path.join(self.result_dir, name + ".log")
        else:
            filename = os.path.join(self.result_dir, name)

        if os.path.exists(filename):
            os.remove(filename)
        return logzero.setup_logger(name=name,
                                    logfile=filename,
                                    formatter=log_fmt)

    def create_tb_writer(self):
        # remove previous tensorboard results
        files = glob.glob(os.path.join(self.result_dir,
                                       'events.out.tfevents*'))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                raise RuntimeError("Error while removing file {}".format(f))
        writer = SummaryWriter(self.result_dir)
        return writer

    def save_config(self, config):
        with open(os.path.join(self.result_dir, "config.yml"), "w") as fp:
            yaml.dump(config, fp)
