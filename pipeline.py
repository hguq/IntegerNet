from cfg import exec_cfg

configs = ["mnist_bit4"]
if __name__ == "__main__":
    for config in configs:
        exec_cfg(f"cfg/{config}.json")
