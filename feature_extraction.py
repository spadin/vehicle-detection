from image import rgb_to_colorspace, rgb_image
from options import image_options, hog_options, color_histogram_options, spatial_binning_options
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import cv2
import numpy as np


def spatial_binning_features(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()

def color_histogram_features(image, bins=32, range=(0., 1.)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:, :, 0], bins=bins, range=range)
    channel2_hist = np.histogram(image[:, :, 1], bins=bins, range=range)
    channel3_hist = np.histogram(image[:, :, 2], bins=bins, range=range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def hog_features(image,
                 channel_index="ALL",
                 orientations=9,
                 pixels_per_cell=(2, 2),
                 cells_per_block=(2, 2)):
    """Helper function for getting HOG features with default parameters for this vehicle detection project"""
    _, _, num_channels = image.shape

    if channel_index == "ALL":
        features = []
        for channel in range(num_channels):
            features.append(
                hog(image[:, :, channel],
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualise=False,
                    feature_vector=True))
        features = np.ravel(features)
    else:
        features = hog(image[:, :, channel_index],
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       visualise=False,
                       feature_vector=True)

    return features


def extract_features_from_list(filepaths,
                               include_hog=True,
                               include_color_histogram=True,
                               include_spatial_binning=True,
                               image_options=image_options,
                               hog_options=hog_options,
                               color_histogram_options=color_histogram_options,
                               spatial_binning_options=spatial_binning_options):
    features = []
    for filepath in tqdm(filepaths, desc="extracting features from list"):
        image = rgb_image(filepath)
        features.append(extract_image_features(image,
                                               include_hog=include_hog,
                                               include_color_histogram=include_color_histogram,
                                               include_spatial_binning=include_spatial_binning,
                                               image_options=image_options,
                                               hog_options=hog_options,
                                               color_histogram_options=color_histogram_options,
                                               spatial_binning_options=spatial_binning_options))

    return features


def extract_image_features(image,
                           include_hog=True,
                           include_color_histogram=True,
                           include_spatial_binning=True,
                           image_options=image_options,
                           hog_options=hog_options,
                           color_histogram_options=color_histogram_options,
                           spatial_binning_options=spatial_binning_options):
    """Extract HOG features for a specific color channel or all channels after converting to a color space"""
    image = rgb_to_colorspace(image, **image_options)
    features = []

    if include_hog: features.append(hog_features(image, **hog_options))
    if include_color_histogram: features.append(color_histogram_features(image, **color_histogram_options))
    if include_spatial_binning: features.append(spatial_binning_features(image, **spatial_binning_options))

    return np.concatenate(features)

def scale_features(features):
    features = np.array(features).astype(np.float64)
    return MinMaxScaler(feature_range=(-1.,1.)).fit_transform(features)


if __name__ == "__main__":
    pass
    # from image import rgb_image
    # from image import rgb_to_gray
    # import matplotlib.pyplot as plt
    # import glob
    # image = rgb_image("./data/vehicles/KITTI_extracted/1.png")
    # gray_image = rgb_to_gray(image)
    # vehicles = glob.glob("./data/vehicles/**/*.png")
    # features = extract_features_from__list(vehicles[0:5])
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
