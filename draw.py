import cv2
import numpy as np

def draw_bounding_boxes(image, bounding_boxes, create_copy=False):
    if create_copy:
        image = np.copy(image)

    for (start, end) in bounding_boxes:
        cv2.rectangle(image, start, end, (0, 0, 255), 4)

    return image
