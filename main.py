import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Program entry point.
    :return: None
    """
    image_directory = "dataset/Training/png/"
    image_name = "001-lighthouse.png"
    img = mpimg.imread(image_directory + image_name)
    plt.imshow(img)
    plt.title(image_name)
    plt.show()

    kernel = np.ones((5, 5))/25  # blur kernel
    conv_img = convolution(img, kernel)
    plt.imshow(conv_img)
    plt.title("Convoluted Image")
    plt.show()


def convolution(image, kernel):
    """
    Outputs an image of the same size as the original. Treats areas outside of the image as zero. Uses images from the
    Training dataset to test the function with a range of kernels: blurs of different sizes and edge detectors in
    different directions.
    :param image: original image
    :param kernel: the kernel used for convolution
    :return: the convoluted image
    """
    img_height = image.shape[0]
    img_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    print("img_height:{}\nimg_width: {}\nkernel_height: {}\nkernel_width: {}".format(
        img_height, img_width, kernel_height, kernel_width)
    )

    convoluted_img = np.zeros((img_height, img_width))

    # loop through each pixel in the image
    for i in range(img_height):
        for j in range(img_width):
            # print("i:{} j:{}".format(i, j))
            accumulator = 0
            # loop through each cell in the kernel
            for m in range(kernel_height):
                for n in range(kernel_width):
                    # todo perform convolution calculations here
                    pass

    return convoluted_img


if __name__ == "__main__":
    main()
