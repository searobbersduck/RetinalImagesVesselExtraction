import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0,10, (256, 256))/10

for i in range(10):
    plt.figure(1)
    plt.subplot(1,3,1)
    # plt.cla()
    img = np.random.randint(0, 10, (256, 256)) / 10
    plt.imshow(img, cmap='hot', interpolation='nearest')
    plt.pause(0.2)
    # plt.show(0.5)
