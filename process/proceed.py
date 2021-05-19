from dataset import get_loader
from process import test_once, train_one_epoch
import importlib
import json
import torch
import math
import copy
import os


def proceed():
    """
    Continue to train
    """
    # Load and parse a json file to a dict "d"
    d = json.loads(open("config.json").read())

    print("********** continue to train **********")
    # Construct model from "net.py"
    model_module = importlib.import_module("net")
    exec(f"model=model_module." + d["MODEL"].split('.')[-1])
    net = eval("model")(**d["MODEL_PARAM"]).cuda()
    # Load weight from "state.pth"
    net.load_state_dict(torch.load("state.pth"))
    # Convert "net" to parallel
    net = torch.nn.DataParallel(net)

    # Get loader from "d"
    train_loader, test_loader = get_loader(d["DATASET"], d["BATCH_SIZE"])

    # Make a optimizer "opt"
    opt = torch.optim.Adam(net.parameters(), lr=d["LR"])

    # Calculate decay rate,
    # make the first epoch with learning rate LR,
    # and last epoch with learning LR_FIN;
    print(f"Suggested start learning rate:{d['LR']:.3e}")
    print(f"Suggested end learning rate:{d['LR_FIN']:.3e}")
    start_lr = float(input("Please input start learning rate:"))
    end_lr = float(input("Please input end learning rate:"))
    n_epoch = int(input("Please input epochs to continue training:"))

    # decay_rate = math.pow(d["LR_FIN"] / d["LR"], 1. / (d["EPOCHS"] - 1))
    decay_rate = math.pow(end_lr / start_lr, 1. / (n_epoch - 1))
    # Make a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, decay_rate) \
        if d["USE_LR_SCH"] else None

    best_state = copy.deepcopy(net.module.state_dict())
    # best_top1_rate, best_top5_rate = test_once(net, test_loader, batch_size=d["BATCH_SIZE"], epoch=0)
    # get best record from this folder name
    dir_str = os.getcwd().split("\\")[-1]
    best_top1_rate, best_top5_rate = map(lambda x: float(x) / 100, dir_str.split("_")[-2:])

    # Log
    log = []

    # Use exception handling to catch keyboard interruption and save model.
    try:
        for e in range(n_epoch):
            # train one epoch
            train_top1_rate, train_top5_rate = train_one_epoch(model=net,
                                                               dataloader=train_loader,
                                                               opt=opt,
                                                               batch_size=d["BATCH_SIZE"],
                                                               epoch=e,
                                                               lr_sch=lr_scheduler)
            # then test accuracy
            top1_rate, top5_rate = test_once(model=net,
                                             dataloader=test_loader,
                                             batch_size=d["BATCH_SIZE"],
                                             epoch=e)
            # Add log
            log.append(f"Epoch {e}: train {train_top1_rate * 100:2.2f} {train_top5_rate * 100:2.2f}\n")
            log.append(f"Epoch {e}: test  {top1_rate * 100:2.2f} {top5_rate * 100:2.2f}\n")
            cur_lr = opt.state_dict()['param_groups'][0]['lr']
            log.append(f"Epoch {e}: LR    {cur_lr:.3e}\n\n")
            # Update best model state
            if top1_rate > best_top1_rate:
                best_top1_rate = top1_rate
                best_top5_rate = top5_rate
                # If model is parallel, only save the module state dict.
                if isinstance(net, torch.nn.DataParallel):
                    best_state = copy.deepcopy(net.module.state_dict())
                else:
                    best_state = copy.deepcopy(net.state_dict())
    # If keyboard interrupt, save current best state before exiting.
    except KeyboardInterrupt:
        print("Interrupt saving...")
    finally:
        save_dir = f"../{d['SAVE_PREFIX']}" \
                   f"_{best_top1_rate * 100:2.2f}" \
                   f"_{best_top5_rate * 100:2.2f}"
        # make a new directory to save state.
        model_file = d["MODEL"].replace(".", "/") + ".py"
        os.system(f"mkdir {save_dir}")
        # Copy model file
        os.system(f"cp net.py {save_dir}/net.py")
        # Copy layer folder
        os.system(f"cp layer/ {save_dir} -r")
        # Copy config file
        os.system(f"cp config.json {save_dir}/config.json")
        # Copy continue file
        os.system(f"cp proceed.py {save_dir}/proceed.py")
        # Copy replay file
        os.system(f"cp replay.py {save_dir}/replay.py")
        # Copy raw replay file
        os.system(f"cp raw.py {save_dir}/raw.py")
        # Save state dict
        torch.save(best_state, save_dir + "/state.pth")
        # Save net structure with state dict
        net.module.load_state_dict(best_state)
        torch.save(net.module, save_dir + "/net.pth")
        # Save log
        with open(f"{save_dir}/log.txt", "w") as f:
            for line in log:
                f.write(line)


if __name__ == "__main__":
    proceed()
