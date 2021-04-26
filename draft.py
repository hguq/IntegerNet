from dataset import get_mnist
import numpy as np

train_loader, test_loader = get_mnist("C:/projects/datasets", batch_size=10000)
if __name__ == "__main__":
    train_x = np.zeros([0, 1, 28, 28])
    train_y = np.zeros([0])
    for x, y in train_loader:
        train_x = np.concatenate([train_x, x], 0)
        train_y = np.concatenate([train_y, y], 0)

    test_x = np.zeros([0, 1, 28, 28])
    test_y = np.zeros([0])
    for x, y in test_loader:
        test_x = np.concatenate([test_x, x], 0)
        test_y = np.concatenate([test_y, y], 0)

    print(train_x.shape, test_x.shape)
