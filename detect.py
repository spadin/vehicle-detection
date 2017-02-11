import time

from model import Model
from image import rgb_image, resize_image, copy_image, rgb_to_ycrcb, scale_image
from feature_extraction import combine_features, combine_hog_features, hog_features_for_channel
from feature_extraction import color_histogram_features, spatial_binning_features
from options import hog_options
from sliding_windows import sliding_window_list
from draw import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label

def create_windows():
    windows = []

    windows.append(sliding_window_list(start=(720, 400),
                                       end=(1280, 584),
                                       scale=1,
                                       pixels_per_step=(16, 16)))

    windows.append(sliding_window_list(start=(720, 400),
                                       end=(1280, 642),
                                       scale=2,
                                       pixels_per_step=(32, 32)))

    return np.vstack(windows)

def hog_features_at_scale(image, scale):
    image = scale_image(image, scale)
    image = image[400//scale:,720//scale:,]

    hog_channel_1 = np.array(hog_features_for_channel(image, 0, feature_vector=False))
    hog_channel_2 = np.array(hog_features_for_channel(image, 1, feature_vector=False))
    hog_channel_3 = np.array(hog_features_for_channel(image, 2, feature_vector=False))

    return hog_channel_1, hog_channel_2, hog_channel_3

def hog_features_for_slice_at_scale(scaled_hogs, scale, start, end, hog_options=hog_options):
    pixels_per_cell_x, pixels_per_cell_y = hog_options["pixels_per_cell"]
    cells_per_block_x, cells_per_block_y = hog_options["cells_per_block"]

    start_x, start_y = start
    end_x, end_y = end

    start_x = start_x - 720
    start_y = start_y - 400

    end_x = end_x - 720
    end_y = end_y - 400

    channel_1_features, channel_2_features, channel_3_features = scaled_hogs[scale]

    start_x = np.int((start_x / scale // pixels_per_cell_x) - (cells_per_block_x - 1)) + 1
    start_y = np.int((start_y / scale // pixels_per_cell_y) - (cells_per_block_y - 1)) + 1

    end_x = np.int((end_x / scale // pixels_per_cell_x) - (cells_per_block_x - 1))
    end_y = np.int((end_y / scale // pixels_per_cell_y) - (cells_per_block_y - 1))

    channel_1_slice = channel_1_features[start_y:end_y, start_x:end_x]
    channel_2_slice = channel_2_features[start_y:end_y, start_x:end_x]
    channel_3_slice = channel_3_features[start_y:end_y, start_x:end_x]

    return channel_1_slice, channel_2_slice, channel_3_slice

def detect(image, model, windows):
    bounding_boxes = []
    ycrcb_image = rgb_to_ycrcb(image)

    scaled_hogs = {}
    scales = [1, 2]

    for scale in scales:
        scaled_hogs[scale] = hog_features_at_scale(ycrcb_image, scale)

    for ((start_x, start_y), (end_x, end_y), scale) in windows:
        slice_x = slice(start_x, end_x)
        slice_y = slice(start_y, end_y)

        image_slice = ycrcb_image[slice_y, slice_x]
        image_slice = resize_image(image_slice)

        hog_slices = hog_features_for_slice_at_scale(scaled_hogs, scale, (start_x, start_y), (end_x, end_y))
        hog_channel_1_slice, hog_channel_2_slice, hog_channel_3_slice = hog_slices

        # hog_channel_1_slice = np.array(hog_features_for_channel(image_slice, 0, feature_vector=False))
        # hog_channel_2_slice = np.array(hog_features_for_channel(image_slice, 1, feature_vector=False))
        # hog_channel_3_slice = np.array(hog_features_for_channel(image_slice, 2, feature_vector=False))

        hog_features = combine_hog_features(
            np.ravel(hog_channel_1_slice),
            np.ravel(hog_channel_2_slice),
            np.ravel(hog_channel_3_slice),
            feature_vector=True
        )

        features = combine_features(hog_features,
                                    color_histogram_features(image_slice),
                                    spatial_binning_features(image_slice))

        prediction = model.predict(features)
        if prediction == 1:
            bounding_boxes.append(((start_x, start_y), (end_x, end_y)))

    return bounding_boxes

def add_heat(heatmap, bounding_boxes):
    for ((start_x, start_y), (end_x, end_y)) in bounding_boxes:
        slice_x = slice(start_x, end_x)
        slice_y = slice(start_y, end_y)
        heatmap[slice_y, slice_x] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def labeled_bounding_boxes(image, labels):
    bounding_boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bounding_boxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    return bounding_boxes

class Detection:
    def __init__(self, model, windows=[]):
        self.model = model
        self.windows = windows
        self.heat_history = []

    def detect(self, image):
        image = copy_image(image)
        bounding_boxes = detect(image, self.model, self.windows)
        heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
        heatmap = add_heat(heatmap, bounding_boxes)

        self.heat_history.append(heatmap)
        self.heat_history = self.heat_history[-3:]

        heat = np.vstack(self.heat_history)
        heat = apply_threshold(heat, threshold=1)

        labels = label(heat)
        bounding_boxes = labeled_bounding_boxes(image, labels)
        image = draw_bounding_boxes(image, bounding_boxes)
        return image

if __name__ == "__main__":
    image = rgb_image("./data/test_images/test1.jpg")
    model = Model()
    windows = create_windows()

    print("Number of windows {}".format(len(windows)))

    # windows = []
    # windows = np.vstack(windows)

    detection = Detection(model, windows)
    image = detection.detect(image)
    # image = draw_bounding_boxes(image, windows)

    plt.imshow(image)
    plt.show()
