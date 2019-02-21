import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def main():
    image_path = "dataset/Training/png/001-lighthouse.png"
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
