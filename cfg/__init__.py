"""
Config implementation
"""
import copy
import os
import json
import threading
import torch
import math
import model
from dataset import get_loader
from process import train_one_epoch, test_once


# This function will execute a specified config file.
# 'override' allows parameter overriding
def exec_cfg(config_path: str, override=None):
    """
    Do a task specified by config file.
    :param override: some extra config parameter, or override parameter
    :param config_path: a config file to be executed
    """
    # Load and parse a json file to a dict "d"
    d = json.loads(open(config_path).read())
    # Override all parameters provided by 'override'
    if override:
        for key, value in override.items():
            d[key] = value

    print("**********" + config_path + "**********")
    # Make a model "net"
    net = eval(d["MODEL"])(**d["MODEL_PARAM"]).cuda()
    # If "state" is specified in kwargs, load state
    if "state_file" in d.keys():
        net.load_state_dict(torch.load(override["state_file"]))
    # Convert "net" to parallel
    net = torch.nn.DataParallel(net)

    # Get loader from "d"
    train_loader, test_loader = get_loader(d["DATASET"], d["BATCH_SIZE"])

    # Make a optimizer "opt"
    opt = torch.optim.Adam(net.parameters(), lr=d["LR"])

    # Calculate decay rate,
    # make the first epoch with learning rate LR,
    # and last epoch with learning LR_FIN;
    decay_rate = math.pow(d["LR_FIN"] / d["LR"], 1. / (d["EPOCHS"] - 1))
    # Make a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, decay_rate) if d["USE_LR_SCH"] else None

    # Initialize iterative variables
    best_state = None
    best_top1_rate = 0
    best_top5_rate = 0

    # Log
    log = []

    # Use exception handling to catch keyboard interruption and save model.
    try:
        for e in range(d["EPOCHS"]):
            # train one epoch
            train_top1_rate, train_top5_rate = train_one_epoch(net=net,
                                                               dataloader=train_loader,
                                                               opt=opt,
                                                               batch_size=d["BATCH_SIZE"],
                                                               epoch=e,
                                                               lr_sch=lr_scheduler)
            # then test accuracy
            test_top1_rate, test_top5_rate = test_once(net=net,
                                                       dataloader=test_loader,
                                                       batch_size=d["BATCH_SIZE"],
                                                       epoch=e)
            # Add log
            cur_lr = opt.state_dict()['param_groups'][0]['lr']
            log.append(f"Epoch {e}: train {train_top1_rate * 100:2.2f} {train_top5_rate * 100:2.2f}\n")
            log.append(f"Epoch {e}: test  {test_top1_rate * 100:2.2f} {test_top5_rate * 100:2.2f}\n")
            log.append(f"Epoch {e}: LR    {cur_lr:.3e}\n\n")
            # Update best model state
            if test_top1_rate > best_top1_rate:
                best_top1_rate = test_top1_rate
                best_top5_rate = test_top5_rate
                # If model is parallel, only save the module state dict.
                if isinstance(net, torch.nn.DataParallel):
                    best_state = copy.deepcopy(net.module.state_dict())
                else:
                    best_state = copy.deepcopy(net.state_dict())
    # If keyboard interrupt, save current best state before exiting.
    except KeyboardInterrupt:
        print("Interrupt saving...")
    finally:
        if "save_dir" not in d:
            save_dir = "save"  # in project root dir save
        else:
            save_dir = d["save_dir"]
        save_dir += f"/{d['SAVE_PREFIX']}" \
                    f"_{best_top1_rate * 100:2.2f}" \
                    f"_{best_top5_rate * 100:2.2f}"

        if "proj_dir" not in d:
            proj_dir = "."  # current folding is project root dir ( when running pipeline )
        else:
            proj_dir = d["proj_dir"]

        # make a new directory to save state.
        os.system(f"mkdir {save_dir}")
        # Copy model file
        model_file = d["MODEL"].replace(".", "/") + ".py"
        os.system(f"cp {proj_dir}/{model_file} {save_dir}/net.py")
        # Copy layer folder
        os.system(f"cp {proj_dir}/layer/ {save_dir} -r")
        # Copy config file
        os.system(f"cp {config_path} {save_dir}/config.json")
        # Copy resume file
        os.system(f"cp {proj_dir}/process/resume.py {save_dir}/resume.py")
        # Copy replay file
        os.system(f"cp {proj_dir}/process/replay.py {save_dir}/replay.py")
        # Copy raw replay file
        os.system(f"cp {proj_dir}/process/raw.py {save_dir}/raw.py")
        # Save state dict
        # If interrupted in first epoch, save current state
        if best_state is None:
            if isinstance(net, torch.nn.DataParallel):
                best_state = copy.deepcopy(net.module.state_dict())
            else:
                best_state = copy.deepcopy(net.state_dict())
        torch.save(best_state, save_dir + "/state.pth")
        # Save net structure with state dict
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(best_state)
            torch.save(net.module, save_dir + "/net.pth")
        else:
            net.load_state_dict(best_state)
            torch.save(net, save_dir + "/net.pth")
        # Save log
        with open(f"{save_dir}/log.txt", "w") as f:
            for line in log:
                f.write(line)


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
