"""
Config implementation
"""
import copy
import os
import shutil
import json
import threading
import torch
import math
import model

from dataset import get_loader
from process import train_one_epoch, test_once


def exec_cfg(config_file):
    """
    Do a task specified by config file.
    :param config_file: a config file to be executed
    """
    d = json.loads(open(config_file).read())
    print("**********" + config_file + "**********")

    train_loader, test_loader = get_loader(d["DATASET"], d["BATCH_SIZE"])
    net = eval(d["MODEL"])(**d["MODEL_PARAM"]).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=d["LR"])

    decay_rate = math.pow(d["LR_FIN"] / d["LR"], 1. / (d["EPOCHS"] - 1))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, decay_rate) \
        if d["USE_LR_SCH"] else None

    best_state = None
    best_top1_rate = 0
    best_top5_rate = 0

    try:
        for e in range(d["EPOCHS"]):
            train_one_epoch(model=net,
                            dataloader=train_loader,
                            opt=opt,
                            batch_size=d["BATCH_SIZE"],
                            epoch=e,
                            lr_sch=lr_scheduler)
            top1_rate, top5_rate = test_once(model=net,
                                             dataloader=test_loader,
                                             batch_size=d["BATCH_SIZE"],
                                             epoch=e)
            if top1_rate > best_top1_rate:
                best_top1_rate = top1_rate
                best_top5_rate = top5_rate
                best_state = copy.deepcopy(net.state_dict())
    except KeyboardInterrupt:
        print("Interrupt saving...")
    finally:
        save_dir = f"save/{d['SAVE_PREFIX']}" \
                   f"_{best_top1_rate * 100:2.2f}" \
                   f"_{best_top5_rate * 100:2.2f}"
        # os.mkdir(save_dir)
        model_file = d["MODEL"].replace(".", "/") + ".py"
        os.system(f"mkdir {save_dir}")
        # Copy model file
        os.system(f"cp {model_file} {save_dir}/net.py")
        # Copy layer folder
        os.system(f"cp layer/ {save_dir} -r")
        # Copy config file
        os.system(f"cp {config_file} {save_dir}/config.json")
        # Copy replay file
        os.system(f"cp process/replay.py {save_dir}/replay.py")

        # Save state dict
        torch.save(best_state, save_dir + "/state.pth")


class TaskQueueThread(threading.Thread):
    """
    A class uses to implement multi-threading training
    """

    def __init__(self, task_queue, task_fn):
        super().__init__()
        self.task_queue = task_queue
        self.task_fn = task_fn

    def run(self):
        """
        execute task by calling task_fn
        """
        for task in self.task_queue:
            self.task_fn(task)


def exec_multi_cfg(config_files):
    """
    execute configs in a multi-threading manner.
    :param config_files: config files
    """
    thread_pool = []
    for config_file in config_files:
        t = TaskQueueThread(config_file, exec_cfg)
        t.start()
        thread_pool.append(t)

    for t in thread_pool:
        t.join()
