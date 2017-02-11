import cv2
import numpy as np
import matplotlib.image as mpimg

def rgb_image(filepath, filetype="jpg"):
    image = mpimg.imread(filepath)

    if filetype == "png":
        image = (image * 255).astype(np.uint8)

    return image

def copy_image(image):
    return np.copy(image)

def resize_image(image, size=(64, 64)):
    return cv2.resize(image, size)

def rgb_to_colorspace(image, color_space="RGB"):
    color_conversion_code = {
        "RGB": None,
        "Gray": cv2.COLOR_RGB2GRAY,
        "HSV": cv2.COLOR_RGB2HSV,
        "HLS": cv2.COLOR_RGB2HLS,
        "LUV": cv2.COLOR_RGB2LUV,
        "LAB": cv2.COLOR_RGB2LAB,
        "YUV":  cv2.COLOR_RGB2YUV,
        "YCrCb": cv2.COLOR_RGB2YCrCb,
    }[color_space]

    if color_conversion_code is not None and color_space != "gray":
        image = cv2.cvtColor(image, color_conversion_code)

    return image

def rgb_to_gray(image):
    return rgb_to_colorspace(image, "Gray")

def rgb_to_hsv(image):
    return rgb_to_colorspace(image, "HSV")

def rgb_to_hls(image):
    return rgb_to_colorspace(image, "HLS")

def rgb_to_luv(image):
    return rgb_to_colorspace(image, "LUV")

def rgb_to_lab(image):
    return rgb_to_colorspace(image, "LAB")

def rgb_to_yuv(image):
    return rgb_to_colorspace(image, "YUV")

def rgb_to_ycrcb(image):
    return rgb_to_colorspace(image, "YCrCb")

if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # filepath = "./data/vehicles/KITTI_extracted/2.png"
    # image = rgb_image(filepath)
    #
    # hsv_image = rgb_to_hsv(image)
    # f, (ax1, ax2) = plt.subplots(1, 2)
    # f.tight_layout()
    #
    # ax1.imshow(hsv_image)
    # ax2.imshow(image)
    # plt.show()