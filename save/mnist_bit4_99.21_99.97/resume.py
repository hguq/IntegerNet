import json
from cfg import exec_cfg


def resume():
    """
    Continue to train
    """
    d = json.load(open("./config.json", "r"))
    try:
        lr = float(input(f"Default LR     is {d['LR']:.3e}, please input LR:"))
        lr_fin = float(input(f"Default LR_FIN is {d['LR_FIN']:.3e}, please input LR_FIN:"))
        epochs = int(input(f"Default EPOCHS is {d['EPOCHS']}, please input EPOCHS:"))
    except ValueError:
        print("Input error, using default")
        lr = d["LR"]
        lr_fin = d["LR_FIN"]
        epochs = d["EPOCHS"]
    print("********** continue to train **********")
    exec_cfg("./config.json",
             {
                 "state_file": "./state.pth",
                 "save_dir": "..",
                 "proj_dir": "../..",
                 "LR": lr,
                 "LR_FIN": lr_fin,
                 "EPOCH": epochs
             })


if __name__ == "__main__":
    resume()
