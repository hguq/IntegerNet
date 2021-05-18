from cfg import exec_cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
configs = ["imagenet_resnet_bit4"]
if __name__ == "__main__":
    for config in configs:
        exec_cfg(f"cfg/{config}.json")
