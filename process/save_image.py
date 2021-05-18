from dataset import get_loader
import numpy as npy
from matplotlib import pyplot as plt

batch_size = 512
train_loader, test_loader = get_loader("mnist", batch_size=batch_size)

f = open("../image_list.txt", "w")


def save_images(images, labels, batch_ind):
    for i in range(batch_size):
        image = images[i]
        label = labels[i]
        image = npy.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:  # only 1 channel
            # image = np.concatenate([image] * 3, axis=2)
            # plt.imsave(f"test_images/{batch_size * batch_ind + i:04d}.{label.item()}.bmp", image / 255)
            f.write(f"{batch_size * batch_ind + i:04d}.{label.item()}.bmp\n")


if __name__ == "__main__":
    # for batch_ind, (images, labels) in enumerate(test_loader):
    #    save_images(images, labels, batch_ind)
    # x = plt.imread("test_images/0000.7.bmp")
    # print(x.shape)
    images, labels = next(iter(test_loader))
    image = images[0]
    for i in range(28):
        for j in range(28):
            print("%03d " % round(image[0, i, j].item()), end='')
        print("")
