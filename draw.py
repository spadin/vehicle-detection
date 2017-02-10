import cv2
import numpy as np

def draw_bounding_boxes(image, bounding_boxes):
    for (start, end) in bounding_boxes:
        cv2.rectangle(image, start, end, (0, 0, 255), 4)

    return image
