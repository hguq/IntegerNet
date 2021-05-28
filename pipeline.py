from cfg import exec_cfg
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# configs = ["imagenet_vgg", "imagenet_vgg_bit4", "imagenet_vgg_bit8"]
configs = ["imagenet_vgg_bit4"]
# override = {"EPOCHS": 2}
if __name__ == "__main__":
    for config in configs:
        exec_cfg(f"cfg/{config}.json")
