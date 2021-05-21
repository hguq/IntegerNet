from dataset import get_loader
from process import test_once
import importlib
import json
import torch


def replay():
    """
    Replay saved model in current directory.
    There should be config.json, model.py, state.pth, layer.
    """
    d = json.loads(open("config.json").read())
    train_loader, test_loader = get_loader(d["DATASET"], d["BATCH_SIZE"])
    model_module = importlib.import_module("net")
    exec(f"model=model_module." + d["MODEL"].split('.')[-1])
    net = eval("model")(**d["MODEL_PARAM"]).cuda()
    net.load_state_dict(torch.load("state.pth"))
    test_once(net, test_loader, batch_size=d["BATCH_SIZE"], epoch=0)


if __name__ == "__main__":
    replay()
