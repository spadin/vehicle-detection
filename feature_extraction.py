from image import rgb_to_colorspace, rgb_image, resize_image
from options import image_options, hog_options, color_histogram_options, spatial_binning_options
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import numpy as np


def spatial_binning_features(image, size=(32, 32)):
    """Gathers all the image pixel information into a 1-D array"""
    return resize_image(image, size).ravel()

def color_histogram_features(image, bins=32):
    """Computes the color histogram features"""
    channel_1_histogram = np.histogram(image[:, :, 0], bins=bins)
    channel_2_histogram = np.histogram(image[:, :, 1], bins=bins)
    channel_3_histogram = np.histogram(image[:, :, 2], bins=bins)

    return np.concatenate((
        channel_1_histogram[0],
        channel_2_histogram[0],
        channel_3_histogram[0]))

def hog_features(image, feature_vector=True, hog_options=hog_options):
    """Helper function for getting HOG features with default parameters for this vehicle detection project"""
    return combine_hog_features(hog_features_for_channel(image, 0, feature_vector, **hog_options),
                                hog_features_for_channel(image, 1, feature_vector, **hog_options),
                                hog_features_for_channel(image, 2, feature_vector, **hog_options),
                                feature_vector=feature_vector)

def combine_hog_features(channel_1, channel_2, channel_3, feature_vector):
    if feature_vector == True:
        return np.ravel([channel_1, channel_2, channel_3])
    else:
        return [channel_1, channel_2, channel_3]


def hog_features_for_channel(image,
                             channel,
                             feature_vector,
                             orientations=9,
                             pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2)):
    return hog(image[:, :, channel],
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               feature_vector=feature_vector,
               visualise=False)


def extract_features_from_list(file_paths,
                               file_type="jpg",
                               image_options=image_options,
                               hog_options=hog_options,
                               color_histogram_options=color_histogram_options,
                               spatial_binning_options=spatial_binning_options):
    features = []
    for file_path in file_paths:
        image = rgb_image(file_path, file_type)
        features.append(extract_image_features(image,
                                               image_options=image_options,
                                               hog_options=hog_options,
                                               color_histogram_options=color_histogram_options,
                                               spatial_binning_options=spatial_binning_options))

    return features


def extract_image_features(image,
                           image_options=image_options,
                           hog_options=hog_options,
                           color_histogram_options=color_histogram_options,
                           spatial_binning_options=spatial_binning_options):
    image = rgb_to_colorspace(image, **image_options)

    return combine_features(hog_features(image, hog_options=hog_options),
                            color_histogram_features(image, **color_histogram_options),
                            spatial_binning_features(image, **spatial_binning_options))

def combine_features(hog_features, color_histogram_features, spatial_binning_features):
    return np.concatenate([hog_features, color_histogram_features, spatial_binning_features])


def scale_training_features(features):
    features = np.array(features).astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    return features, scaler

if __name__ == "__main__":
    pass
