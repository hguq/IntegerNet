from layer.quantize import get_normal_weight
from matplotlib import pyplot as plt

x = get_normal_weight((100000,), 4)

plt.hist(x.cpu().numpy(),bins=1000)
plt.show()
