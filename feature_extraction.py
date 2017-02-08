from skimage.feature import hog
from image import rgb_to_colorspace
import numpy as np

def hog_features(image,
                 orientations=9,
                 pixels_per_cell=(2, 2),
                 cells_per_block=(2, 2),
                 visualise=False,
                 feature_vector=False):
    """Helper function for getting HOG features with default parameters for this vehicle detection project"""
    return hog(image,
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               visualise=visualise,
               feature_vector=feature_vector)

def extract_features(image,
                     channel_index="ALL",
                     color_space="RGB",
                     orientations=9,
                     pixels_per_cell=2,
                     cells_per_block=2):
    """Extract HOG features for a specific color channel or all channels after converting to a color space"""
    image = np.copy(image)
    image = rgb_to_colorspace(image, color_space)
    _, _, num_channels = image.shape

    if channel_index == "ALL":
        features = []
        for channel in range(num_channels):
            features.append(
                hog_features(image[:, :, channel],
                             orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             visualise=False,
                             feature_vector=True))
        features = np.ravel(features)
    else:
        features = hog_features(image[:, :, channel_index],
                                orientations=orientations,
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,
                                visualise=False,
                                feature_vector=True)

    return features

if __name__ == "__main__":
    pass
    # from image import rgb_image
    # from image import rgb_to_gray
    # import matplotlib.pyplot as plt

    # image = rgb_image("./data/vehicles/KITTI_extracted/1.png")
    # gray_image = rgb_to_gray(image)

    # hog_features, hog_image = hog_features(gray_image, visualise=True)
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # f.tight_layout()
    #
    # ax1.set_title("Original image")
    # ax1.imshow(image)
    #
    # ax2.set_title("Gray image")
    # ax2.imshow(gray_image, cmap="gray")
    #
    # ax3.set_title("HOG features")
    # ax3.imshow(hog_image, cmap="gray")
    #
    # plt.show()
