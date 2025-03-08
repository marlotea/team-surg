import json
import os
from os.path import join
import torch
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger, WandbLogger 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


LIGHTNING_CKPT_PATH = 'lightning_logs/version_0/checkpoints/'
LIGHTNING_TB_PATH = 'lightning_logs/version_0/'
LIGHTNING_METRICS_PATH = 'lightning_logs/version_0/metrics.csv'


def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name, "ckpts")


def get_ckpt_callback(save_path, exp_name, filename):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    return ModelCheckpoint(dirpath=ckpt_dir,
                           filename=filename,
                           save_top_k=1,
                           verbose=True,
                           monitor='val_f1',
                           mode='max')


def get_early_stop_callback(patience=10):
    return EarlyStopping(monitor='val_f1', #TODO - monitor metric
                         patience=patience,
                         verbose=True,
                         mode='max')


def get_logger(save_path="", exp_name="", test=False, wandb_hps=None, 
                project="", log_exp_name=""):
    #Create logging params and save paths 
    if not wandb_hps:
        wandb_hps = {} 
    if len(log_exp_name) == 0:
        log_exp_name = exp_name 
    exp_dir = os.path.join(save_path, exp_name)
    wandb_test_save_path = os.path.join(exp_dir, "wandb_test")
    if not os.path.exists(wandb_test_save_path):
        os.mkdir(wandb_test_save_path)

    #Create logger objects
    tt_logger = CSVLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version="0")
    wandb_logger = WandbLogger(project=project, name=log_exp_name, save_dir=wandb_test_save_path)
    if dist.is_initialized() and dist.get_rank() != 0:
        wandb_logger.experiment.config.update(wandb_hps, allow_val_change=True)
    if test:
        return wandb_logger
    else:
        return [tt_logger, wandb_logger]

class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)


def init_exp_folder(args):
    save_dir = os.path.abspath(args.get("save_dir"))
    exp_name = args.get("exp_name")
    exp_path = join(save_dir, exp_name)
    exp_metrics_path = join(exp_path, "metrics.csv")
    exp_tb_path = join(exp_path, "tb")
    global_tb_path = args.get("tb_path")
    global_tb_exp_path = join(global_tb_path, exp_name)

    # init exp path
    if os.path.exists(exp_path):
        raise FileExistsError(f"Experiment path [{exp_path}] already exists!")
    os.makedirs(exp_path, exist_ok=True)

    os.makedirs(global_tb_path, exist_ok=True)
    if os.path.exists(global_tb_exp_path):
        raise FileExistsError(f"Experiment exists in the global "
                              f"Tensorboard path [{global_tb_path}]!")
    os.makedirs(global_tb_path, exist_ok=True)

    # dump hyper-parameters/arguments
    with open(join(save_dir, exp_name, "args.json"), "w") as f:
        json.dump(args, f)

    # ln -s for metrics
    os.symlink(join(exp_path, LIGHTNING_METRICS_PATH),
               exp_metrics_path)

    # ln -s for tb
    os.symlink(join(exp_path, LIGHTNING_TB_PATH), exp_tb_path)
    os.symlink(exp_tb_path, global_tb_exp_path)
